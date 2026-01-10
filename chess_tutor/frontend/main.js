const API_BASE = window.API_BASE || 'http://localhost:8000';
try{ console.log('[BUILD] chess_tutor main.js active'); }catch(_){}
try{ window.__RENDER_PLY_COUNT__ = window.__RENDER_PLY_COUNT__ || 0; }catch(_){}

async function fetchJSON(url, opts={}){
  const res = await fetch(url, {headers:{'Content-Type':'application/json'}, ...opts});
  if(!res.ok){
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.json();
}
// --- RAG Chat ---
const CHAT = { history: [] };
let CHAT_INCLUDE_GAMES = true; // toggle via checkbox
let CHAT_INCLUDE_PRINCIPLES = true;
let CURRENT_PRINCIPLE_ID = null; // optional context from deep-link

// Build a text template summarizing the current position
function buildPositionSummaryTemplate(){
  try{
    const fen = (window.__LAST_FEN__ || '').trim();
    const stm = (fen.split(' ')[1]||'').toLowerCase();
    const sideToMove = stm === 'w' ? 'White' : (stm === 'b' ? 'Black' : '');
    const sideMoved = sideToMove === 'White' ? 'Black' : (sideToMove === 'Black' ? 'White' : '');
    const sanList = (typeof currentMoves !== 'undefined' && Array.isArray(currentMoves)) ? currentMoves : [];
    let lastMove = '-';
    try{
      if(typeof currentPly !== 'undefined' && currentPly>0 && sanList.length >= currentPly){
        lastMove = sanList[currentPly-1] || '-';
      }
    }catch(_){/* ignore */}
    // Collect principles
    const details = Array.isArray(window.__LAST_PRINCIPLE_DETAILS__) ? window.__LAST_PRINCIPLE_DETAILS__ : [];
    const overlays = window.__LAST_OVERLAYS__ || { arrows: [], highlights: [] };
    const canonical = (id)=>{
      if(!id) return '';
      const map = {
        'AttackedPieces':'Attacked Pieces',
        'IsolatedPawns':'Isolated Pawns',
        'UndefendedPieces':'Undefended Pieces',
        'DoubledPawns':'Doubled Pawns',
        'PassedPawns':'Passed Pawns',
        'PinnedPieces':'Pinned Pieces',
        'HangingPieces':'Hanging Pieces',
        'KingCheck':'King in Check',
        'Checkmate':'Checkmate',
        'SpaceAdvantage':'Space Advantage',
      };
      return map[id] || id;
    };
    const byIdSide = new Map(); // key: id|side -> { id, side, squares: Set }
    for(const d of details){
      if(!d || !d.id) continue;
      const id = canonical(d.id);
      const side = (d.side||'').toUpperCase();
      const key = id + '|' + side;
      const entry = byIdSide.get(key) || { id, side, squares: new Set() };
      const sqs = Array.isArray(d.squares) ? d.squares : [];
      for(const s of sqs){ if(s) entry.squares.add(String(s)); }
      byIdSide.set(key, entry);
    }
    // Fallback: group overlay highlights by principle if no details
    if(byIdSide.size === 0 && overlays && Array.isArray(overlays.highlights)){
      for(const h of overlays.highlights){
        const id = canonical(h && h.principle);
        if(!id) continue;
        const key = id + '|';
        const entry = byIdSide.get(key) || { id, side: '', squares: new Set() };
        if(h && h.square) entry.squares.add(String(h.square));
        byIdSide.set(key, entry);
      }
    }
    // Compose bullets grouped deterministically
    const bullets = [];
    const preferredOrder = ['Attacked Pieces','Undefended Pieces','Isolated Pawns','Doubled Pawns','Passed Pawns','Pinned Pieces','Hanging Pieces','King in Check','Checkmate','Space Advantage'];
    const entries = Array.from(byIdSide.values());
    entries.sort((a,b)=>{
      const ia = preferredOrder.indexOf(a.id); const ib = preferredOrder.indexOf(b.id);
      if(ia!==-1 || ib!==-1){ return (ia===-1?999:ia) - (ib===-1?999:ib); }
      if(a.id!==b.id) return a.id.localeCompare(b.id);
      return (a.side||'').localeCompare(b.side||'');
    });
    for(const e of entries){
      const sideLabel = e.side === 'W' ? 'White' : (e.side === 'B' ? 'Black' : '');
      const sqs = Array.from(e.squares);
      const sqStr = sqs.length ? sqs.join(', ') : '-';
      const prefix = sideLabel ? `${e.id} for ${sideLabel}` : e.id;
      bullets.push(`    - ${prefix}: ${sqStr}`);
    }
    if(bullets.length === 0){ bullets.push('    - (none detected)'); }
    const tpl = [
      `* Move: ${lastMove}`,
      `* FEN String: ${fen || '-'}`,
      `* Side that has moved: ${sideMoved || '-'}`,
      `* Side to move: ${sideToMove || '-'}`,
      `* Principles Highlights:`,
      ...bullets
    ].join('\n');
    return tpl;
  }catch(e){
    return '* Move: -\n* FEN String: -\n* Side that has moved: -\n* Side to move: -\n* Principles Highlights:\n    - (error building summary)';
  }
}

function showPositionSummaryModal(text, onConfirm){
  const modal = document.getElementById('position-summary-modal');
  const body = document.getElementById('position-summary-text');
  const btnOk = document.getElementById('psm-confirm');
  const btnCancel = document.getElementById('psm-cancel');
  if(!modal || !body || !btnOk || !btnCancel){ onConfirm && onConfirm(); return; }
  body.textContent = text || '';
  modal.classList.remove('hidden');
  const clean = ()=>{ modal.classList.add('hidden'); btnOk.onclick=null; btnCancel.onclick=null; };
  btnOk.onclick = ()=>{ try{ onConfirm && onConfirm(); }finally{ clean(); } };
  btnCancel.onclick = ()=>{ clean(); };
}

function safeAnnotate(text, moves){
  try{
    if(typeof annotateTextWithMoves === 'function'){
      return annotateTextWithMoves(text, moves||[]);
    }
  }catch(_){/* noop */}
  try{
    if(typeof escapeHTML === 'function') return escapeHTML(text||'');
  }catch(_){/* noop */}
  return (text||'');
}

function pushChat(role, content, opts={}){
  CHAT.history.push({ role, content });
  const box = document.getElementById('chat-messages');
  if(!box) return;
  const div = document.createElement('div');
  div.className = 'chat-line';
  const who = role === 'user' ? 'You' : 'Tutor';
  let body = content || '';
  if(role === 'assistant'){
    // annotate moves if we have a current game
    body = safeAnnotate(body, (typeof currentMoves!=='undefined'?currentMoves:[]) );
  }else{
    body = (typeof escapeHTML==='function') ? escapeHTML(body) : body;
  }
  div.innerHTML = `<b>${who}:</b> ${body}`;
  // Optional sources list
  if(role === 'assistant' && Array.isArray(opts.sources) && opts.sources.length){
    const srcWrap = document.createElement('div');
    srcWrap.className = 'chat-sources';
    srcWrap.style.margin = '6px 0 4px 0';
    opts.sources.forEach((s,i)=>{
      const item = document.createElement('div');
      item.className = 'chat-source-item small';
      const meta = s.meta || {}; const labelParts = [];
      if(meta.id) labelParts.push(meta.id);
      if(meta.type) labelParts.push(meta.type);
      if(meta.page !== undefined) labelParts.push('p.'+meta.page);
      const label = labelParts.join(' • ') || `source ${i+1}`;
      item.innerHTML = `<span class="src-label">[${i+1}] ${escapeHTML(label)}:</span> ${escapeHTML((s.snippet||'').slice(0,160))}`;
      srcWrap.appendChild(item);
    });
    div.appendChild(srcWrap);
  }
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function sendChat(){
  const inp = document.getElementById('chat-input');
  if(!inp) return;
  const q = (inp.value||'').trim();
  if(!q) return;
  inp.value = '';
  pushChat('user', q);
  try{
    // show loading spinner line
    const loadingId = 'chat-loading';
    const box = document.getElementById('chat-messages');
    if(box){
      const loadDiv = document.createElement('div');
      loadDiv.className = 'chat-line small';
      loadDiv.id = loadingId;
      loadDiv.textContent = 'Thinking…';
      box.appendChild(loadDiv); box.scrollTop = box.scrollHeight;
    }
    const payload = { messages: CHAT.history.slice(-12), include_games: CHAT_INCLUDE_GAMES, include_principles: CHAT_INCLUDE_PRINCIPLES };
    try{
      if(typeof currentGameId !== 'undefined' && currentGameId){
        payload.game_id = currentGameId;
        if(typeof currentPly !== 'undefined') payload.ply = currentPly;
      }
      if(CURRENT_PRINCIPLE_ID){ payload.principle_id = CURRENT_PRINCIPLE_ID; }
    }catch(_){/* ignore */}
    const res = await fetchJSON(`${API_BASE}/chat`, { method:'POST', body: JSON.stringify(payload) });
    let answer = res.answer || '';
    // remove loading
    const ld = document.getElementById(loadingId); if(ld) ld.remove();
    pushChat('assistant', answer, { sources: res.sources });
  }catch(e){
    const ld = document.getElementById('chat-loading'); if(ld) ld.remove();
    pushChat('assistant', `Sorry, I couldn't fetch an answer (${e.message||e}).`);
  }
}


// Read deep-link params
const URL_PARAMS = new URLSearchParams(window.location.search);

// Load and render games list
async function loadGames(){
  const listEl = document.getElementById('games');
  if(listEl) listEl.innerHTML = '<li class="small loading">Loading games...</li>';
  try {
    const games = await fetchJSON(`${API_BASE}/games`);
    window.__ALL_GAMES__ = games || [];
    // Seed PDF set with all games if both sets are empty (initial state)
    if (window.__PDF_GAMES__ && window.__PDF_GAMES__.size === 0 && window.__PGN_GAMES__ && window.__PGN_GAMES__.size === 0) {
      for (const g of window.__ALL_GAMES__) { window.__PDF_GAMES__.add(g.id); }
    }
    // Reclassify based on id pattern (pgn/pdf) to fix mixed tabs
    if(typeof reclassifyAllGames === 'function') reclassifyAllGames();
    renderSourceGameLists();
  } catch (err) {
    if (listEl) listEl.innerHTML = `<li class="error">Failed to load games: ${err.message}</li>`;
    // Ensure UI lists render even on failure
    renderSourceGameLists();
  }
}

function renderGameList(filter=''){
  const ul = document.getElementById('games');
  if(!ul) return;
  ul.innerHTML = '';
  const q = (filter||'').toLowerCase();
  const games = (window.__ALL_GAMES__||[]).filter(g=>!q || `${g.id}`.toLowerCase().includes(q));
  if(!games.length){
    ul.innerHTML = '<li class="small">No games</li>';
    return;
  }
  games.forEach((g, i)=>{
    const li = document.createElement('li');
    li.textContent = `[${i+1}] ${g.id} (len=${g.length})`;
    li.onclick = ()=>selectGame(g.id, li);
    if(g.id === currentGameId) li.classList.add('selected');
    ul.appendChild(li);
  });
  }
  function filterPdfGames(q) {
      const pdfUl = document.getElementById('pdf-games-list');
      if (!pdfUl) return;
      const query = (q || '').toLowerCase();
      pdfUl.querySelectorAll('li').forEach(li => {
          if (li.classList.contains('small')) return;
          const id = (li.textContent || '').toLowerCase();
          li.style.display = !query || id.includes(query) ? '' : 'none';
      });
  }

// Utilities for annotating explanation text with clickable SAN tokens
function escapeHTML(s){
  return s.replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));
}
function escapeRegExp(s){
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
function annotateTextWithMoves(text, moves){
  if(!text) return '';
  let out = '';
  let pos = 0;
  for(let i=0;i<moves.length;i++){
    const tok = moves[i];
    const re = new RegExp(escapeRegExp(tok));
    const slice = text.slice(pos);
    const m = re.exec(slice);
    if(!m){ continue; }
    const start = pos + m.index;
    const end = start + tok.length;
    out += escapeHTML(text.slice(pos, start));
    out += `<span class="gt-move" data-ply="${i+1}">${escapeHTML(tok)}</span>`;
    pos = end;
  }
  out += escapeHTML(text.slice(pos));
  return out;
}
function updateActiveTextMove(ply){
  document.querySelectorAll('.gt-move').forEach(el=>{
    el.classList.toggle('active', Number(el.dataset.ply) === ply);
  });
}

const PIECE_TO_UNICODE = {
  'P':'♙','N':'♘','B':'♗','R':'♖','Q':'♕','K':'♔',
  'p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚'
};

// board overlay arrow helpers
const FILES = ['a','b','c','d','e','f','g','h'];
// Custom FEN retrieval removed; use engine-provided FEN only
function squareToCoord(sq){
  const file = FILES.indexOf(sq[0]);
  const rank = 8 - parseInt(sq[1], 10);
  const cell = 100/8; // percent space in viewBox units
  const x = (file+0.5)*cell;
  const y = (rank+0.5)*cell;
  return {x,y};
}
function clearOverlay(){
  const svg = document.getElementById('board-overlay');
  if(svg) svg.innerHTML = '';
}
function drawArrow(fromSq, toSq, color='#10b981'){
  try{ console.log('[DRAW_ARROW_ENTRY]', { from: fromSq, to: toSq, color }); }catch(_){}
  const svg = document.getElementById('board-overlay');
  if(!svg || !fromSq || !toSq){
    try{ console.log('[DRAW_ARROW_SKIP]', { hasSvg: !!svg, hasFrom: !!fromSq, hasTo: !!toSq }); }catch(_){}
    return;
  }
  try{ console.log('[DRAW_ARROW]', { from: String(fromSq), to: String(toSq), color: String(color) }); }catch(_){}
  const {x: x1, y: y1} = squareToCoord(fromSq);
  const {x: x2, y: y2} = squareToCoord(toSq);
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.hypot(dx, dy) || 1;
  const cell = 100/8;
  // Shorten at destination so shaft is visible (knight moves fix)
  const shortenEnd = Math.min((cell/2) - 1.2, dist - 0.5);
  const sx2 = x2 - (dx/dist) * shortenEnd;
  const sy2 = y2 - (dy/dist) * shortenEnd;
  // Optionally offset start for attack (red) arrows so piece not obscured
  let sx1 = x1;
  let sy1 = y1;
  if(color.toLowerCase() === '#ff1a1a'){
    const shortenStart = Math.min((cell/2) - 1.2, dist - 0.5);
    sx1 = x1 + (dx/dist) * shortenStart;
    sy1 = y1 + (dy/dist) * shortenStart;
  }
  let defs = svg.querySelector('defs');
  if(!defs){
    defs = document.createElementNS('http://www.w3.org/2000/svg','defs');
    svg.appendChild(defs);
  }
  // Separate marker per color to avoid head recolor side-effects
  const markerId = 'arrowhead-' + color.replace(/[^a-zA-Z0-9]/g,'');
  let marker = svg.querySelector('#'+markerId);
  if(!marker){
    marker = document.createElementNS('http://www.w3.org/2000/svg','marker');
    marker.setAttribute('id', markerId);
    marker.setAttribute('markerUnits','userSpaceOnUse');
    marker.setAttribute('markerWidth','4');
    marker.setAttribute('markerHeight','6');
    marker.setAttribute('refX','3');
    marker.setAttribute('refY','3');
    marker.setAttribute('orient','auto');
    marker.setAttribute('viewBox','0 0 8 8');
    const path = document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d','M0,0 L6,3 L0,6 Z');
    path.setAttribute('fill', color);
    marker.appendChild(path);
    defs.appendChild(marker);
  }
  const line = document.createElementNS('http://www.w3.org/2000/svg','line');
  line.setAttribute('x1', sx1);
  line.setAttribute('y1', sy1);
  line.setAttribute('x2', sx2);
  line.setAttribute('y2', sy2);
  line.setAttribute('stroke', color);
  line.setAttribute('stroke-width','1.25');
  line.setAttribute('stroke-linecap','round');
  line.setAttribute('marker-end',`url(#${markerId})`);
  svg.appendChild(line);
}

// Ensure any global calls hit our instrumented implementation
try{
  if(!window.__DRAW_ARROW_WRAPPED__){
    window.__DRAW_ARROW_WRAPPED__ = true;
    const __origDrawArrow = drawArrow;
    window.drawArrow = function(fromSq, toSq, color){
      try{ console.log('[DRAW_ARROW_WRAP]', { from: fromSq, to: toSq, color }); }catch(_){}
      return __origDrawArrow(fromSq, toSq, color);
    };
  }
}catch(_){/* ignore wrapper issues */}

function getArrowEndpoints(a){
  if(!a) return { from:null, to:null };
  const from = a.from || a.source || a.src || a.from_sq || null;
  const to = a.to || a.target || a.dst || a.to_sq || null;
  return { from, to };
}

function getHighlightSquare(h){
  if(!h) return null;
  // Accept common variants
  const sq = h.square || h.sq || h.to || h.target || null;
  if(typeof sq === 'string') return sq;
  return null;
}

function drawCircle(sq, color='#ff1a1a'){
  const svg = document.getElementById('board-overlay');
  if(!svg || !sq) return;
  const {x, y} = squareToCoord(sq);
  const cell = 100/8;
  const r = (cell/2) - 2.5;
  const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('cx', x);
  c.setAttribute('cy', y);
  c.setAttribute('r', r);
  c.setAttribute('stroke', color);
  c.setAttribute('stroke-width', '1.4');
  c.setAttribute('fill', 'none');
  svg.appendChild(c);
}

function drawDoubleCircle(sq, color='#ff1a1a'){
  const svg = document.getElementById('board-overlay');
  if(!svg || !sq) return;
  const {x, y} = squareToCoord(sq);
  const cell = 100/8;
  const r1 = (cell/2) - 2.5;      // outer ring
  const r2 = r1 - 6;              // inner ring (wider gap for visual distinction)
  for(const r of [r1, r2]){
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx', x);
    c.setAttribute('cy', y);
    c.setAttribute('r', r);
    c.setAttribute('stroke', color);
    c.setAttribute('stroke-width', '1.4');
    c.setAttribute('fill', 'none');
    svg.appendChild(c);
  }
}

function drawSquare(sq, color='#ff1a1a'){
  const svg = document.getElementById('board-overlay');
  if(!svg || !sq) return;
  const {x, y} = squareToCoord(sq);
  const cell = 100/8;
  const half = (cell/2) - 2.5; // match circle margin
  const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
  rect.setAttribute('x', (x - half));
  rect.setAttribute('y', (y - half));
  rect.setAttribute('width', (half * 2));
  rect.setAttribute('height', (half * 2));
  rect.setAttribute('stroke', color);
  rect.setAttribute('stroke-width', '1.4');
  rect.setAttribute('stroke-linejoin', 'round');
  rect.setAttribute('fill', 'none');
  svg.appendChild(rect);
}

// Add red border to impacted squares (replaces previous shading)
function highlightSquares(squares){
  // Remove any prior borders
  document.querySelectorAll('.principle-border').forEach(el=>{
    el.classList.remove('principle-border');
  });
  document.querySelectorAll('.principle-border-space').forEach(el=>{
    el.classList.remove('principle-border-space');
  });
  if(!Array.isArray(squares) || !squares.length) return;
  // Validate coordinates: only allow a-h + 1-8
  const coordOk = s => typeof s === 'string' && /^[a-h][1-8]$/.test(s);
  const dedup = new Set();
  const filtered = [];
  for(const h of squares){
    const sq = h && h.square;
    if(coordOk(sq) && !dedup.has(sq)){
      dedup.add(sq);
      filtered.push(h);
    }
  }
  if(!filtered.length) return;
  const ui = document.getElementById('board-ui');
  if(!ui) return;
  const cells = ui.querySelectorAll('.square');
  const map = new Map();
  for(let idx=0; idx<cells.length; idx++){
    const r = Math.floor(idx/8); // rank index top-down
    const f = idx % 8;
    const sq = FILES[f] + (8 - r);
    map.set(sq, cells[idx]);
  }
  for(const h of filtered){
    const sq = h && h.square;
    if(!sq) continue;
    const cell = map.get(sq);
    if(!cell) continue;
    // If tagged specifically as space advantage, use black border class
    if(h && h.principle === 'SpaceAdvantage'){
      cell.classList.add('principle-border-space');
    }else{
      cell.classList.add('principle-border');
    }
  }
}

// Utilities for annotating explanation text with clickable SAN tokens
function escapeHTML(s){
  return s.replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));
}
function escapeRegExp(s){
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function renderFenToGrid(fen){
  const ui = document.getElementById('board-ui');
  ui.innerHTML = '';
  const [placement] = fen.split(' ');
  const rows = placement.split('/');
  for(let r=0; r<8; r++){
    const row = rows[r];
    let file = 0;
    for(const ch of row){
      if(/\d/.test(ch)){
        const count = parseInt(ch, 10);
        for(let i=0;i<count;i++){
          const sq = document.createElement('div');
          sq.className = `square ${(r+file)%2===0? 'dark':'light'}`;
          ui.appendChild(sq);
          file++;
        }
      }else{
        const sq = document.createElement('div');
        sq.className = `square ${(r+file)%2===0? 'dark':'light'}`;
        const glyph = PIECE_TO_UNICODE[ch] || '';
        sq.textContent = glyph;
        if(glyph){
          const isWhite = ch === ch.toUpperCase();
          sq.classList.add(isWhite? 'piece-white' : 'piece-black');
          // Add piece-specific class for targeted styling (e.g., white queen)
          const basePiece = ch.toUpperCase();
          // Use lowercase marker for white queen for CSS (.white-queen)
          if(isWhite && basePiece === 'Q'){
            sq.classList.add('white-queen');
          }
        }
        ui.appendChild(sq);
        file++;
      }
    }
  }
}

function pgnToInlineElements(moves, ply){
  const frag = document.createDocumentFragment();
  for(let i=0;i<moves.length;i+=2){
    const moveNo = Math.floor(i/2)+1;
    const w = moves[i] || '';
    const b = moves[i+1] || '';
    const wIdx = i+1;
    const bIdx = i+2;

    const label = document.createElement('span');
    label.className = 'small';
    label.textContent = `${moveNo}. `;
    frag.appendChild(label);

    const wEl = document.createElement('span');
    wEl.className = 'move' + (ply===wIdx? ' active':'');
    wEl.textContent = w;
    wEl.dataset.ply = String(wIdx);
    frag.appendChild(wEl);

    if(b){
      frag.appendChild(document.createTextNode(' '));
      const bEl = document.createElement('span');
      bEl.className = 'move' + (ply===bIdx? ' active':'');
      bEl.textContent = b;
      bEl.dataset.ply = String(bIdx);
      frag.appendChild(bEl);
    }
    frag.appendChild(document.createTextNode('\n'));
  }
  return frag;
}

let currentGameTags = new Map(); // ply -> [principles]

function renderPGN(moves, ply){
  const pre = document.getElementById('pgn');
  pre.textContent = '';
  pre.appendChild(pgnToInlineElements(moves, ply));
  // augment with principle badges if available
  annotatePGNWithPrinciples();
}

function annotatePGNWithPrinciples(){
  const pre = document.getElementById('pgn');
  if(!pre) return;
  pre.querySelectorAll('.move').forEach(el=>{
    const p = parseInt(el.dataset.ply||'0',10);
    const tags = currentGameTags.get(p) || [];
    // remove any existing badge directly following this move
    const next = el.nextElementSibling;
    if(next && next.classList && next.classList.contains('p-badge')){
      next.remove();
    }
    if(tags.length && window.__SHOW_PGN_TAGS__ !== false){
      const span = document.createElement('span');
      span.className = 'p-badge';
      span.textContent = `  [${tags.slice(0,2).join(', ')}${tags.length>2?'…':''}]`;
      span.title = tags.join(', ');
      span.style.opacity = .85;
      el.insertAdjacentElement('afterend', span);
    }
  });
}

async function fetchGamePrinciples(gameId){
  try{
    const data = await fetchJSON(`${API_BASE}/games/${encodeURIComponent(gameId)}/principles`, { cache: 'no-store' });
    currentGameTags = new Map((data.tags||[]).map(t=>[t.ply, t.principles]));
  }catch(_e){ currentGameTags = new Map(); }
}

let currentGameId = null;
let currentPly = 0;
let currentMoves = [];
let currentLi = null;
let includeRelevantGames = false; // disabled by default per backend flag
let __explainCtl = null; // AbortController for explanation fetches

function setExplanationAnnotated(text){
  const ans = document.getElementById('answer');
  // Keep the header line if present, annotate the rest
  const nl = (text||'').indexOf('\n');
  const header = nl>=0 ? (text||'').slice(0, nl+1) : '';
  const body = nl>=0 ? (text||'').slice(nl+1) : (text||'');
  ans.innerHTML = escapeHTML(header) + annotateTextWithMoves(body, currentMoves);
}

async function selectGame(id, liEl){
  currentGameId = id;
  currentPly = 0;
  // Set title using metadata if available
  try{
    const g = (window.__ALL_GAMES__||[]).find(x=>x && x.id===id);
    if(g){
      const w=(g.white||'').trim(); const b=(g.black||'').trim(); const res=(g.result||'').trim();
      const ev=(g.event||'').trim(); const site=(g.site||'').trim();
      let label = id;
      if(w||b){
        label = `${w||'White'} vs ${b||'Black'}`;
        if(res){ label += ` (${res})`; }
        const loc=[]; if(ev) loc.push(ev); if(site) loc.push(site);
        if(loc.length){ label += ` — ${loc.join(' / ')}`; }
      }
      const tEl = document.getElementById('game-title'); if(tEl) tEl.textContent = label;
    }else{
      document.getElementById('game-title').textContent = id;
    }
  }catch(_){ document.getElementById('game-title').textContent = id; }
  // highlight in source lists
  document.querySelectorAll('.games-sublist li').forEach(el=>{
    el.classList.toggle('selected', (el.textContent||'') === id);
  });
  // fetch all moves
  try{
    const data = await fetchJSON(`${API_BASE}/games/${encodeURIComponent(id)}/moves`, { cache: 'no-store' });
    currentMoves = data.moves || [];
  }catch(e){ currentMoves = []; }
  // fetch per-move principles
  await fetchGamePrinciples(id);
  // fetch explanation based on this game (abort previous, show loading, bust cache)
  const ans = document.getElementById('answer');
  const btn = document.getElementById('toggle-explain');
  ans.textContent = 'Loading explanation...';
  ans.classList.add('collapsed');
  if(btn) btn.textContent = 'Show more';
  try{
    if(__explainCtl) __explainCtl.abort();
    __explainCtl = new AbortController();
    const url = new URL(`${API_BASE}/games/${encodeURIComponent(id)}/explain`);
    if(includeRelevantGames) url.searchParams.set('include_relevant_games', 'true');
    url.searchParams.set('ts', Date.now().toString()); // cache buster
    const ex = await fetchJSON(url.toString(), { signal: __explainCtl.signal, cache: 'no-store' });
    setExplanationAnnotated(ex.text || '');
  }catch(e){
    if(e && e.name === 'AbortError') return; // ignore aborted fetches
    ans.textContent = '';
  }finally{
    __explainCtl = null;
  }
  await renderPly();
}

function renderPrinciples(list, details, overlays){
  const box = document.getElementById('principles');
  if(!box) return;
  box.innerHTML = '';
  // Prefer structured details if provided
  if(Array.isArray(details) && details.length){
    details.forEach(d=>{
      const id = d.id || '';
      const side = d.side || '';
      // Deduplicate squares for compact labels
      const squaresRaw = Array.isArray(d.squares) ? d.squares.filter(Boolean) : [];
      const squares = Array.from(new Set(squaresRaw));
      const captured = Array.isArray(d.captured) ? d.captured.filter(Boolean) : [];
      let label = id;
      if(squares.length){
        // Show up to 4 squares to keep label compact
        const shown = squares.slice(0,4);
        const sideLower = side.toLowerCase();
        const sideName = sideLower === 'w' ? 'White' : (sideLower === 'b' ? 'Black' : '');
        const coordStr = shown.join(',');
        const capPart = captured.length ? ` +${captured.slice(0,6).join(',')}` : '';
        label = sideName ? `${id} ${sideName} (${coordStr}${squares.length>4? ',…':''})${capPart}` : `${id} (${coordStr}${squares.length>4? ',…':''})${capPart}`;
      }else{
        const sideLower = side.toLowerCase();
        const sideName = sideLower === 'w' ? 'White' : (sideLower === 'b' ? 'Black' : '');
        const capPart = captured.length ? ` +${captured.slice(0,6).join(',')}` : '';
        if(sideName) label = `${id} ${sideName}${capPart}`; else label = `${id}${capPart}`;
      }
      const span = document.createElement('span');
      span.className = 'badge';
      span.textContent = label;
      const sideLower = side.toLowerCase();
      const sideName = sideLower === 'w' ? 'White' : (sideLower === 'b' ? 'Black' : '');
      let title = squares.length ? `${id}${sideName? ' '+sideName:''}: ${squares.join(', ')}` : (sideName ? `${id} ${sideName}` : id);
      if(captured.length){
        title += `\nCaptured: ${captured.join(', ')}`;
      }
      span.title = title;
      box.appendChild(span);
    });
    return;
  }
  // Fallback: try to extract squares from overlays if provided
  if(overlays && overlays.highlights && Array.isArray(overlays.highlights) && list && list.length){
    const hlSquares = overlays.highlights.map(h=>h && h.square).filter(Boolean);
    const sqJoined = hlSquares.slice(0,4).join(',');
    (list||[]).forEach(tag=>{
      const span = document.createElement('span');
      span.className = 'badge';
      if(hlSquares.length){
        span.textContent = `${tag} (${sqJoined}${hlSquares.length>4?',…':''})`;
        span.title = `${tag}: ${hlSquares.join(',')}`;
      }else{
        span.textContent = tag;
      }
      box.appendChild(span);
    });
    return;
  }
  // Fallback: plain list of ids
  (list||[]).forEach(tag=>{
    const span = document.createElement('span');
    span.className = 'badge';
    span.textContent = tag;
    box.appendChild(span);
  });
}

let __rendering = false;
async function renderPly(){
  try{ console.log('[RENDER_PLY_ENTRY]', { currentGameId, rendering: __rendering }); }catch(_){}
  if(!currentGameId || __rendering) return;
  __rendering = true;
  try{ window.__RENDER_PLY_COUNT__ = (window.__RENDER_PLY_COUNT__||0) + 1; console.log('[RENDER_PLY_COUNT]', window.__RENDER_PLY_COUNT__); }catch(_){}
  // --- Minimal robust overlays/principles reprocessing ---
  // 1. Always clear overlays and principle borders
  // --- DEBUG LOGGING ---
  console.log('[DEBUG] renderPly called:', {currentPly, currentGameId});
  clearOverlay();
  document.querySelectorAll('.principle-border,.principle-border-space').forEach(el=>{
    el.classList.remove('principle-border','principle-border-space');
  });
  try{
    // 2. Always fetch overlays/principles for current ply
    const plyReq = {game_id: currentGameId, ply: currentPly};
    console.log('[DEBUG] Fetching /ply:', plyReq);
    const data = await fetchJSON(`${API_BASE}/ply`, {
      method:'POST',
      body: JSON.stringify(plyReq)
    });
    console.log('[DEBUG] /ply response:', {
      ply: currentPly,
      fen: data.fen,
      overlays: data.overlays,
      principle_details: data.principle_details
    });
    // Halt on checkmate: disable Next button if principle includes Checkmate
    try{
      const nextBtn = document.getElementById('next');
      const isMate = Array.isArray(data.principles) && data.principles.includes('Checkmate');
      if(nextBtn){ nextBtn.disabled = !!isMate; }
    }catch(_){/* ignore UI halt errors */}
    // Use fen_after for board rendering, fen_before for overlays/principles if present
    const fenForBoard = data.fen_after || data.fen;
    const fenForOverlay = data.fen_before || data.fen;
    if (fenForBoard) {
      window.__LAST_FEN__ = fenForBoard;
    }
    document.getElementById('ply').value = currentPly;
    if(fenForBoard){
      renderFenToGrid(fenForBoard);
      const fenText = document.getElementById('fen-text');
      console.log('[FEN DEBUG] Setting FEN string:', fenForBoard, 'Element:', fenText);
      if(fenText) fenText.textContent = `FEN: ${fenForBoard}`;
    }
    const sanMoves = currentMoves && currentMoves.length ? currentMoves : (data.san || []);
    renderPGN(sanMoves, currentPly);
    updateActiveTextMove(currentPly);
    // Use overlays_after and overlays_before for new backend
    window.__LAST_OVERLAYS__ = data.overlays_after || data.overlays || null;
    window.__LAST_OVERLAYS_BEFORE__ = data.overlays_before || null;
    // Suppress 'Tactics' badge unless a multi-move engine line is available
    const hasMulti = Array.isArray(data.variations) && data.variations.some(v=>Array.isArray(v.line) && v.line.length>=2 && Array.isArray(v.fens) && v.fens.length>=2);
    const filteredDetails = Array.isArray(data.principle_details) ? data.principle_details.filter(d=>{
      if(!d || !d.id) return true;
      if(String(d.id).toLowerCase()==='tactics') return !!hasMulti;
      return true;
    }) : [];
    window.__LAST_PRINCIPLE_DETAILS__ = filteredDetails;
    renderPrinciples(data.principles || [], filteredDetails, window.__LAST_OVERLAYS__);
    renderVariations(data.variations || []);
    // Immediate toggle for Show Tactics using unified tactics helper
    try{
      const btn = document.getElementById('show-tactics');
      const countEl = document.getElementById('tactics-count');
      const dbgEl = document.getElementById('tactics-debug');
      const ts = tacticsStateFromPly(data);
      console.log('[TACTICS_DEBUG]', { principles: data.principles||[], details: data.principle_details||[], highlights: (data.overlays_after&&data.overlays_after.highlights)||[], count: ts.count, enabled: ts.enabled });
      if(countEl){ countEl.textContent = `(${ts.count})`; }
      if(dbgEl){ dbgEl.textContent = `count=${ts.count}`; }
      if(btn){ setBtnEnabled(btn, ts.enabled); }
    }catch(_){/* ignore */}
    // Update Candidates button state based on availability
    try{
      await updateCandidatesState();
    }catch(_){/* ignore */}
    // If Candidates tab is active, refresh its list; otherwise clear stale content
    try{
      const candPanel = document.getElementById('tab-moves-candidates');
      const inCandidateMode = !!window.__CANDIDATE_MODE__;
      if(candPanel && candPanel.classList.contains('active') && !inCandidateMode){
        await showCandidates();
      }else if(!inCandidateMode){
        const cl = document.getElementById('candidate-list');
        if(cl) cl.innerHTML = '';
      }
    }catch(_){/* ignore */}
    try{
      if(!CURRENT_PRINCIPLE_ID && Array.isArray(data.principles) && data.principles.length){
        CURRENT_PRINCIPLE_ID = data.principles[0];
      }
    }catch(_){/* ignore */}
    // Draw main move arrow early to avoid later blocks interfering
    try{
      let prevFen = fenForOverlay || null;
      // Fallback: if fen_before missing or identical to current, fetch prev ply
      if((!prevFen || (fenForBoard && prevFen === fenForBoard)) && currentPly>0){
        try{
          const prev = await fetchJSON(`${API_BASE}/ply`, { method:'POST', body: JSON.stringify({game_id: currentGameId, ply: currentPly-1}) });
          prevFen = prev.fen_after || prev.fen || prevFen;
          console.log('[MOVE_ARROW_FALLBACK_PREV_PLY]', { plyMinus1: currentPly-1, prevFen });
        }catch(_){/* ignore fetch fallback errors */}
      }
      try{ console.log('[MOVE_ARROW_ATTEMPT]', { ply: currentPly, hasPrev: !!prevFen, hasCurr: !!fenForBoard }); }catch(_){/* ignore */}
      if(prevFen && fenForBoard){
        try{ console.log('[MOVE_ARROW_PREVFEN]', prevFen); }catch(_){/* ignore */}
        try{ console.log('[MOVE_ARROW_CURRFEN]', fenForBoard); }catch(_){/* ignore */}
        const move = inferMoveFromFen(prevFen, fenForBoard);
        try{ console.log('[MOVE_ARROW_RESULT]', { ply: currentPly, move }); }catch(_){/* ignore */}
        if(move && move.from && move.to){ drawArrow(move.from, move.to, '#22c55e'); }
      }
    }catch(err){ console.warn('[OVERLAYS] move arrow draw failed', err); }

    // 3. Only draw overlays/highlights if overlays are ON
    if(window.__SHOW_OVERLAYS__){
      try{
        console.log('[OVERLAYS] enabled; drawing overlays');
          // Draw overlays/arrows for current ply (use overlays_after)
          const ovAfter = (data.overlays_after || data.overlays || { arrows: [], highlights: [] });
        if(ovAfter && Array.isArray(ovAfter.arrows)){
          console.log('[OVERLAYS] after arrows:', ovAfter.arrows.length);
          try{ console.log('[OVERLAYS] after sample:', JSON.stringify(ovAfter.arrows.slice(0,3))); }catch(_){/* ignore */}
          for(const a of ovAfter.arrows){
            try{
              const color = (a && a.color) ? a.color.toLowerCase() : '';
              const ep = getArrowEndpoints(a);
              if(color === '#ff1a1a'){
                  // After arrows target the current board state
                  const adj = retargetAttackArrow({from:ep.from, to:ep.to}, fenForBoard);
                if(adj) drawArrow(adj.from, adj.to, a.color || '#ff1a1a');
              }else{
                if(ep && ep.from && ep.to) drawArrow(ep.from, ep.to, a.color || '#eab308');
              }
            }catch(err){ console.warn('[OVERLAYS] error drawing after arrow', err, a); }
          }
        }
        // Optionally draw overlays/arrows for previous mover (use overlays_before)
        const ovBefore = (data.overlays_before || { arrows: [], highlights: [] });
        if(ovBefore && Array.isArray(ovBefore.arrows)){
          console.log('[OVERLAYS] before arrows:', ovBefore.arrows.length);
          try{ console.log('[OVERLAYS] before sample:', JSON.stringify(ovBefore.arrows.slice(0,3))); }catch(_){/* ignore */}
          for(const a of ovBefore.arrows){
            try{
              const color = (a && a.color) ? a.color.toLowerCase() : '';
              const ep = getArrowEndpoints(a);
              if(color === '#ff1a1a'){
                  // Before arrows target the previous board state
                  const adj = retargetAttackArrow({from:ep.from, to:ep.to}, fenForOverlay);
                if(adj) drawArrow(adj.from, adj.to, a.color || '#ff1a1a');
              }else{
                if(ep && ep.from && ep.to) drawArrow(ep.from, ep.to, a.color || '#eab308');
              }
            }catch(err){ console.warn('[OVERLAYS] error drawing before arrow', err, a); }
          }
        }
        // (removed) on-screen overlays debug text
      }catch(err){ console.error('[OVERLAYS] drawing failure', err); }
      // Augment: synthesize a red attack arrow if the last move
      // placed a piece onto a ray of an opposing slider.
      try{
        if(currentPly>0){
          const prev = await fetchJSON(`${API_BASE}/ply`, {
            method:'POST',
            body: JSON.stringify({game_id: currentGameId, ply: currentPly-1})
          });
          const last = inferMoveFromFen(prev.fen_after || prev.fen, fenForBoard);
          if(last && last.to){
            const board = buildBoardMapFromFen(fenForBoard);
            const stm = (fenForBoard.split(' ')[1]||'').toLowerCase(); // 'w' or 'b'
            const movedWasWhite = stm === 'b'; // if black to move now, white just moved
            const attackerColor = movedWasWhite ? 'black' : 'white';
            const attackers = findSlidingAttackersForTarget(last.to, board, attackerColor);
            // Avoid duplicates: gather existing red arrows
            const existing = new Set();
            const gather = (ov)=>{ if(ov && Array.isArray(ov.arrows)){ for(const a of ov.arrows){ if((a.color||'').toLowerCase()==='#ff1a1a' && a.from && a.to){ existing.add(`${a.from}->${a.to}`); } } } };
            gather(data.overlays_after); gather(data.overlays_before);
            for(const from of attackers){
              const key = `${from}->${last.to}`;
              if(!existing.has(key)){
                const aPc = board[from];
                const tPc = board[last.to];
                if(aPc && tPc && isOppositeColors(aPc, tPc)){
                  drawArrow(from, last.to, '#ff1a1a');
                }
              }
            }
          }
        }
      }catch(_){/* ignore augmentation errors */}
      // Strict: derive highlights from both after and before overlays
      const impactedSet = new Map(); // key -> {square, principle}
      const addSquare = (sq, principle) => {
        if(!sq || !/^[a-h][1-8]$/.test(sq)) return;
        if(!impactedSet.has(sq)) impactedSet.set(sq, {square: sq, principle: principle||'AttackedPieces'});
      };
      const afterArrows = (ovAfter && Array.isArray(ovAfter.arrows)) ? ovAfter.arrows : [];
      for(const a of afterArrows){ const ep = getArrowEndpoints(a); if(ep){ addSquare(ep.from, 'AttackedPieces'); addSquare(ep.to, 'AttackedPieces'); } }
      const afterHighlights = (ovAfter && Array.isArray(ovAfter.highlights)) ? ovAfter.highlights : [];
      for(const h of afterHighlights){ addSquare(getHighlightSquare(h), h && h.principle); }
      const beforeArrows = (ovBefore && Array.isArray(ovBefore.arrows)) ? ovBefore.arrows : [];
      for(const a of beforeArrows){ const ep = getArrowEndpoints(a); if(ep){ addSquare(ep.from, 'AttackedPieces'); addSquare(ep.to, 'AttackedPieces'); } }
      const beforeHighlights = (ovBefore && Array.isArray(ovBefore.highlights)) ? ovBefore.highlights : [];
      for(const h of beforeHighlights){ addSquare(getHighlightSquare(h), h && h.principle); }
      const impacted = Array.from(impactedSet.values());
      console.log('[OVERLAYS] impacted squares count:', impacted.length);
      // Fallback: if impacted is empty but we have after highlights, draw overlay squares directly
      if(!impacted.length && afterHighlights.length){
        for(const h of afterHighlights){
          const sq = getHighlightSquare(h);
          if(sq) drawSquare(sq, (h && h.color) || '#eab308');
        }
      }
      highlightSquares(impacted);
      // King check/checkmate: draw circle(s) on the attacked king square(s)
      try{
        const squaresMap = new Map(); // sq -> { isMate: boolean }
        const pdet = Array.isArray(data.principle_details) ? data.principle_details : [];
        for(const d of pdet){
          if(!d || !d.id) continue;
          if(d.id === 'KingCheck' || d.id === 'Checkmate'){
            const squares = Array.isArray(d.squares) ? d.squares : [];
            for(const raw of squares){
              if(!raw) continue;
              const sq = String(raw).toLowerCase();
              if(!/^[a-h][1-8]$/.test(sq)) continue;
              const prev = squaresMap.get(sq) || { isMate: false };
              prev.isMate = prev.isMate || (d.id === 'Checkmate');
              squaresMap.set(sq, prev);
            }
          }
        }
        // Also consider overlay highlights (if details absent or supplementary)
        if(data.overlays_after && Array.isArray(data.overlays_after.highlights)){
          for(const h of data.overlays_after.highlights){
            const pid = (h && h.principle) || '';
            const raw = (h && h.square) || '';
            if(!raw) continue;
            const sq = String(raw).toLowerCase();
            if(!/^[a-h][1-8]$/.test(sq)) continue;
            if(pid === 'KingCheck' || pid === 'Checkmate'){
              const prev = squaresMap.get(sq) || { isMate: false };
              prev.isMate = prev.isMate || (pid === 'Checkmate');
              squaresMap.set(sq, prev);
            }
          }
        }
        for(const [sq, info] of squaresMap.entries()){
          if(info.isMate){ drawSquare(sq, '#ff1a1a'); } else { drawCircle(sq, '#ff1a1a'); }
        }
      }catch(_){/* ignore */}
    }
    // (moved earlier)

  }catch(_e){} finally{
    // Count already updated via /ply in render flow; avoid redundant call
    __rendering = false;
    console.log('[OVERLAYS] renderPly complete; flag=', window.__SHOW_OVERLAYS__);
  }
}

function inferMoveFromFen(fenPrev, fenCurr){
  if(!fenPrev || !fenCurr) return null;
  // Improved diff: identify all squares where piece disappeared (candidatesFrom) and appeared (candidatesTo)
  const [pPrev] = fenPrev.split(' ');
  const [pCurr] = fenCurr.split(' ');
  const prevArr = expandFenPlacement(pPrev);
  const currArr = expandFenPlacement(pCurr);
  const squares = [];
  for(let r=0;r<8;r++) for(let f=0; f<8; f++) squares.push(FILES[f] + (8-r));
  const disappeared = []; // {piece, square}
  const appeared = [];   // {piece, square}
  for(let i=0;i<64;i++){
    const a = prevArr[i];
    const b = currArr[i];
    if(a!==b){
      if(a!=='.' && b==='.'){
        disappeared.push({piece:a, square:squares[i]});
      }else if(a!=='.' && b!=='.'){
        // capture or promotion: from square has different piece now
        disappeared.push({piece:a, square:squares[i]});
        appeared.push({piece:b, square:squares[i]});
      }else if(a==='.' && b!=='.'){
        appeared.push({piece:b, square:squares[i]});
      }
    }
  }
  if(!disappeared.length || !appeared.length) return null;
  // Attempt to pair by piece type ignoring case (promotion will differ)
  for(const d of disappeared){
    // Try exact piece match first
    let match = appeared.find(a=>a.piece===d.piece);
    if(!match){
      // Try same color different piece (promotion) choose appeared with same color
      const isWhite = d.piece === d.piece.toUpperCase();
      match = appeared.find(a=> (a.piece===a.piece.toUpperCase()) === isWhite);
    }
    if(match){
      return {from:d.square, to:match.square};
    }
  }
  // Fallback: first disappeared to first appeared
  return {from:disappeared[0].square, to:appeared[0].square};
}

function expandFenPlacement(placement){
  const rows = placement.split('/');
  const out = [];
  for(const row of rows){
    for(const ch of row){
      if(/\d/.test(ch)){
        const n = parseInt(ch,10);
        for(let i=0;i<n;i++) out.push('.');
      }else{
        out.push(ch);
      }
    }
  }
  return out;
}

function squareToFileRank(sq){
  const file = FILES.indexOf(sq[0]);
  const rank = parseInt(sq[1],10) - 1; // 0..7 from bottom
  return {file, rank};
}

function buildBoardMapFromFen(fen){
  if(!fen) return {};
  const map = {};
  const [placement] = fen.split(' ');
  const rows = placement.split('/');
  for(let r=0; r<8; r++){
    let file = 0;
    for(const ch of rows[r]){
      if(/\d/.test(ch)){
        file += parseInt(ch,10);
      }else{
        const sq = FILES[file] + (8 - r);
        map[sq] = ch; // piece char
        file++;
      }
    }
  }
  return map;
}

function isSlidingPiece(pc){
  if(!pc) return false;
  const p = pc.toLowerCase();
  return p==='q' || p==='r' || p==='b';
}

function isOppositeColors(a,b){
  if(!a || !b) return false;
  const aw = a===a.toUpperCase();
  const bw = b===b.toUpperCase();
  return aw !== bw;
}

function pathClear(fromSq, toSq, boardMap){
  const {file: f1, rank: r1} = squareToFileRank(fromSq);
  const {file: f2, rank: r2} = squareToFileRank(toSq);
  const df = Math.sign(f2 - f1);
  const dr = Math.sign(r2 - r1);
  // ensure straight or diagonal
  if(!((df===0 && dr!==0) || (dr===0 && df!==0) || (Math.abs(df)===Math.abs(dr) && df!==0))) return false;
  let cf = f1 + df;
  let cr = r1 + dr;
  while(!(cf===f2 && cr===r2)){
    const sq = FILES[cf] + (cr+1);
    if(boardMap[sq]) return false;
    cf += df; cr += dr;
  }
  return true;
}

function shouldDrawAttackArrow(a, fen){
  if(!a || !a.from || !a.to) return false;
  // Only validate sliding path when we can identify a sliding piece at from and a target piece at to
  try{
    const board = buildBoardMapFromFen(fen);
    const attacker = board[a.from];
    const target = board[a.to];
    // Both endpoints must be occupied; no arrows to/from empty squares
    if(!attacker || !target) return false;
    if(!isOppositeColors(attacker, target)) return false;
    if(isSlidingPiece(attacker)){
      return pathClear(a.from, a.to, board);
    }
    // Non-sliding pieces: allow (knight, king, pawn handled by backend)
    return true;
  }catch(_){ return true; }
}

function retargetAttackArrow(a, fen){
  if(!a || !a.from || !a.to) return null;
  try{
    const board = buildBoardMapFromFen(fen);
    const attacker = board[a.from];
    if(!attacker) return null;
    const target = board[a.to];
    const {file: f1, rank: r1} = squareToFileRank(a.from);
    const {file: f2, rank: r2} = squareToFileRank(a.to);
    const df = Math.sign(f2 - f1);
    const dr = Math.sign(r2 - r1);
    // Non-sliding: require occupied opposite-colored target
    if(!isSlidingPiece(attacker)){
      if(target && isOppositeColors(attacker, target)) return {from:a.from, to:a.to};
      return null;
    }
    // Sliding piece: if target is valid and path clear, keep; else retarget to first occupied along ray
    if(target && isOppositeColors(attacker, target) && pathClear(a.from, a.to, board)){
      return {from:a.from, to:a.to};
    }
    // Walk along ray to find first occupied square
    let cf = f1 + df;
    let cr = r1 + dr;
    while(cf>=0 && cf<8 && cr>=0 && cr<8){
      const sq = FILES[cf] + (cr+1);
      if(board[sq]){
        const piece = board[sq];
        if(isOppositeColors(attacker, piece)) return {from:a.from, to:sq};
        return null;
      }
      // stop if we've reached the intended to-square without finding a blocker
      if(cf===f2 && cr===r2) break;
      cf += df; cr += dr;
    }
    return null;
  }catch(_){ return null; }
}

// Infer newly uncovered sliding attacks targeting a specific square.
// Scans rays from the target square and returns attacker squares matching the color.
function findSlidingAttackersForTarget(targetSq, board, attackerColor){
  const attackers = [];
  if(!targetSq || !board[targetSq]) return attackers;
  const targetIsWhite = board[targetSq] === board[targetSq].toUpperCase();
  const wantBlack = attackerColor === 'black';
  // 8 ray directions: diagonals and orthogonals
  const dirs = [
    {df:1, dr:1}, {df:1, dr:-1}, {df:-1, dr:1}, {df:-1, dr:-1},
    {df:1, dr:0}, {df:-1, dr:0}, {df:0, dr:1}, {df:0, dr:-1}
  ];
  const {file: tf, rank: tr} = squareToFileRank(targetSq);
  for(const d of dirs){
    let cf = tf + d.df;
    let cr = tr + d.dr;
    while(cf>=0 && cf<8 && cr>=0 && cr<8){
      const sq = FILES[cf] + (cr+1);
      const pc = board[sq];
      if(pc){
        const pcIsWhite = pc === pc.toUpperCase();
        const pcIsBlack = !pcIsWhite;
        // Must be the requested attacker color and opposite of target color
        if((wantBlack ? pcIsBlack : pcIsWhite) && (pcIsWhite !== targetIsWhite)){
          const p = pc.toLowerCase();
          const isDiag = Math.abs(d.df) === 1 && Math.abs(d.dr) === 1;
          const isOrtho = (Math.abs(d.df) + Math.abs(d.dr)) === 1;
          const ok = (isDiag && (p==='b' || p==='q')) || (isOrtho && (p==='r' || p==='q'));
          if(ok){ attackers.push(sq); }
        }
        break; // first blocker ends the ray
      }
      cf += d.df; cr += d.dr;
    }
  }
  return attackers;
}

function download(filename, text){
  const el = document.createElement('a');
  el.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  el.setAttribute('download', filename);
  el.style.display = 'none';
  document.body.appendChild(el);
  el.click();
  document.body.removeChild(el);
}

function movesToPGN(moves){
  let out = '';
  for(let i=0;i<moves.length;i+=2){
    const moveNo = Math.floor(i/2)+1;
    const w = moves[i] || '';
    const b = moves[i+1] || '';
    out += `${moveNo}. ${w}${b? ' ' + b : ''}\n`;
  }
  return out.trim();
}

// --- Source ingestion (PDF / PGN) ---
window.__PDF_GAMES__ = new Set();
window.__PGN_GAMES__ = new Set();
async function uploadSource(kind){
  const pdfInput = document.getElementById('pdf-input');
  const pgnInput = document.getElementById('pgn-input');
  const sourceNameEl = document.getElementById('source-name');
  const minMovesEl = document.getElementById('min-moves');
  const before = new Set((window.__ALL_GAMES__||[]).map(g=>g.id));
  let fileEl = kind === 'pdf' ? pdfInput : pgnInput;
  if(!fileEl || !fileEl.files || !fileEl.files.length){
    alert(`Select a ${kind.toUpperCase()} file first.`);
    return;
  }
  const file = fileEl.files[0];
  const form = new FormData();
  form.append('file', file);
  form.append('source', sourceNameEl && sourceNameEl.value ? sourceNameEl.value : file.name);
  if(kind === 'pdf'){
    const mv = parseInt((minMovesEl && minMovesEl.value) || '8', 10) || 8;
    form.append('min_moves', String(mv));
  }
  const endpoint = kind === 'pdf' ? '/author/upload_pdf' : '/author/upload_pgn';
  try{
    const res = await fetch(`${API_BASE}${endpoint}`, { method:'POST', body: form });
    if(!res.ok){ throw new Error(await res.text() || res.statusText); }
    const data = await res.json();
    const sumEl = document.getElementById('ingest-summary');
    if(sumEl){
      const entries = Object.entries(data.principles_indexed||{}).filter(e=>e[1]>0).sort((a,b)=>b[1]-a[1]).slice(0,12);
      const issues = Array.isArray(data.issues) && data.issues.length ? `<div class="small warn"><b>Issues:</b> ${escapeHTML(data.issues.join(' | '))}</div>` : '';
      sumEl.innerHTML = `<div class="small"><b>Loaded:</b> ${data.games_added} new game(s) (total ${data.total_games}) from ${escapeHTML(data.source)}.<br/>Principles: ${entries.map(e=>`${e[0]}:${e[1]}`).join(' ') || 'none'}</div>${issues}`;
    }
    await loadGames(); // refresh list
    // Diff new games and classify
    const after = new Set((window.__ALL_GAMES__||[]).map(g=>g.id));
    for(const id of after){
      if(!before.has(id)){
        if(kind==='pdf'){
          window.__PDF_GAMES__.add(id);
          // Ensure exclusivity
          window.__PGN_GAMES__.delete(id);
        } else {
          window.__PGN_GAMES__.add(id);
          // Remove from PDF set if it was misclassified earlier
          window.__PDF_GAMES__.delete(id);
        }
      }
    }
    // Re-run pattern classification to catch any naming-based hints
    reclassifyAllGames();
    renderSourceGameLists();
    updatePrincipleCounts();
  }catch(e){
    alert(`Upload failed: ${e.message||e}`);
  }
}

function renderSourceGameLists(){
  const pgnUl = document.getElementById('pgn-games-list');
  const pdfUl = document.getElementById('pdf-games-list');
  // Build quick lookup map id->game for metadata formatting
  const gameMap = {};
  (window.__ALL_GAMES__||[]).forEach(g=>{ if(g && g.id) gameMap[g.id] = g; });
  function formatLabel(g){
    if(!g) return '';
    const w = (g.white||'').trim();
    const b = (g.black||'').trim();
    const ev = (g.event||'').trim();
    const site = (g.site||'').trim();
    const res = (g.result||'').trim();
    // If we have at least white or black names, prefer metadata label
    if(w || b){
      let core = `${w||'White'} vs ${b||'Black'}`;
      if(res){ core += ` (${res})`; }
      const locParts = [];
      if(ev) locParts.push(ev);
      if(site) locParts.push(site);
      if(locParts.length){ core += ` — ${locParts.join(' / ')}`; }
      return core;
    }
    return g.id; // fallback
  }
  if(pgnUl){
    pgnUl.innerHTML='';
    const ids=[...window.__PGN_GAMES__];
    if(!ids.length){ pgnUl.innerHTML='<li class="small">None yet</li>'; }
    ids.forEach(id=>{
      const li=document.createElement('li');
      const g = gameMap[id];
      li.textContent = g ? formatLabel(g) : id;
      li.title = id; // preserve original id on hover
      li.onclick=()=>selectGame(id, null);
      if(id===currentGameId) li.classList.add('selected');
      pgnUl.appendChild(li);
    });
  }
  if(pdfUl){
    pdfUl.innerHTML='';
    // PDF list excludes any PGN games for exclusivity
    const ids=[...window.__PDF_GAMES__].filter(id=>!window.__PGN_GAMES__.has(id));
    if(!ids.length){ pdfUl.innerHTML='<li class="small">None yet</li>'; }
    ids.forEach(id=>{
      const li=document.createElement('li');
      const g = gameMap[id];
      // Only show metadata if present (PDF extractions may not have player names)
      li.textContent = g && (g.white||g.black) ? formatLabel(g) : id;
      li.title = id;
      li.onclick=()=>selectGame(id, null);
      if(id===currentGameId) li.classList.add('selected');
      pdfUl.appendChild(li);
    });
  }
}

// Reclassification helper: assign games to PGN/PDF sets based on id tokens
function reclassifyAllGames(){
  if(!window.__ALL_GAMES__) return;
  window.__PDF_GAMES__ = window.__PDF_GAMES__ || new Set();
  window.__PGN_GAMES__ = window.__PGN_GAMES__ || new Set();
  for(const g of window.__ALL_GAMES__){
    const idLower = (g.id||'').toLowerCase();
    if(idLower.includes('pgn')){
      window.__PGN_GAMES__.add(g.id);
      window.__PDF_GAMES__.delete(g.id);
    }else if(idLower.includes('pdf')){
      window.__PDF_GAMES__.add(g.id);
      window.__PGN_GAMES__.delete(g.id);
    }
  }
}

async function updatePrincipleCounts(){
  try{
    const rows = await fetchJSON(`${API_BASE}/principles`);
    const wrap = document.getElementById('principle-counts');
    if(wrap){
      wrap.innerHTML = rows.map(r=>`<span class="pc">${escapeHTML(r.id)}:${r.examples}</span>`).join(' ');
    }
  }catch(_){/* ignore */}
}

window.addEventListener('load', async ()=>{
  try{
    await loadGames();
  }catch(err){
    document.getElementById('games').innerHTML = `<li style="color:#f87171">Failed to load games: ${err.message}</li>`;
  }
  updatePrincipleCounts();
  // Tab switching scoped per tab group (.tabs)
  document.querySelectorAll('.tabs .tab-header').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const container = btn.closest('.tabs');
      if(!container) return;
      container.querySelectorAll('.tab-header').forEach(b=>b.classList.remove('active'));
      container.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
      btn.classList.add('active');
      const id = btn.dataset.tab;
      const panel = container.querySelector('#'+id);
      if(panel){ panel.classList.add('active'); }
      // If switching into Candidates tab, refresh button state and list
      if(id === 'tab-moves-candidates'){
        const inCandidateMode = !!window.__CANDIDATE_MODE__;
        try{ updateCandidatesState(); }catch(_){/* ignore */}
        if(!inCandidateMode){
          try{ showCandidates(); }catch(_){/* ignore */}
        }
      }
    });
  });

  // Auto upload when selecting hidden file inputs triggered by dropdown
  const pdfInput = document.getElementById('pdf-input');
  const pgnInput = document.getElementById('pgn-input');
  if(pdfInput){ pdfInput.addEventListener('change', ()=>{ if(pdfInput.files.length) uploadSource('pdf'); }); }
  if(pgnInput){ pgnInput.addEventListener('change', ()=>{ if(pgnInput.files.length) uploadSource('pgn'); }); }
  // Initial render for source game lists (empty state)
  renderSourceGameLists();

  // Deep-link: ?game=<id>&ply=<n>
  const deepGame = URL_PARAMS.get('game');
  const deepPly = parseInt(URL_PARAMS.get('ply')||'0',10);
  const deepPrinciple = URL_PARAMS.get('principle');
  if(deepPrinciple){ CURRENT_PRINCIPLE_ID = deepPrinciple; }
  if(deepGame){
    await selectGame(deepGame, null);
    if(!Number.isNaN(deepPly) && deepPly>=0){
      currentPly = deepPly;
      await renderPly();
    }
    // refresh list to reflect selected item highlight
    renderGameList(document.getElementById('game-search')?.value||'');
    renderSourceGameLists();
  }

  // search handler
  const search = document.getElementById('game-search');
  if(search){
    search.addEventListener('input', (e)=>{
      filterPdfGames(e.target.value);
    });
  }

  // Sync moves list height to board area (board-wrap) once DOM laid out
  function syncMovesHeight(){
    const boardWrap = document.getElementById('board-wrap');
    const movesCol = document.querySelector('.moves-column');
    if(boardWrap && movesCol){
      const h = boardWrap.offsetHeight; // includes board + file labels
      movesCol.style.maxHeight = h + 'px';
      movesCol.style.height = h + 'px';
    }
  }
  syncMovesHeight();
  window.addEventListener('resize', ()=>{ syncMovesHeight(); });

  // Dropdown menu bindings for Games suboptions
  const ddView = document.getElementById('dd-view-games');
  const ddPdf = document.getElementById('dd-upload-pdf');
  const ddPgn = document.getElementById('dd-upload-pgn');
  if(ddView){ ddView.addEventListener('click', ()=>{ document.getElementById('games')?.scrollIntoView({behavior:'smooth'}); }); }
  if(ddPdf){ ddPdf.addEventListener('click', ()=>{ document.getElementById('pdf-input')?.click(); }); }
  if(ddPgn){ ddPgn.addEventListener('click', ()=>{ document.getElementById('pgn-input')?.click(); }); }

  // toggle explanation collapse
  const toggle = document.getElementById('toggle-explain');
  const ans = document.getElementById('answer');
  // chat wiring
  const chatBtn = document.getElementById('chat-send');
  const chatInput = document.getElementById('chat-input');
  if(chatBtn){ chatBtn.addEventListener('click', sendChat); }
  if(chatInput){ chatInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter') sendChat(); }); }
  // Add dynamic chat toggles if placeholder container exists
  const chatPanel = document.getElementById('chat-panel');
  if(chatPanel && !document.getElementById('chat-toggles')){
    const tog = document.createElement('div');
    tog.id = 'chat-toggles';
    tog.className = 'small';
    tog.style.marginTop = '4px';
    tog.innerHTML = `
      <label style="margin-right:12px"><input type="checkbox" id="chat-inc-games" checked /> Include games</label>
      <label><input type="checkbox" id="chat-inc-principles" checked /> Include principles</label>
      <button id="chat-explain-position" class="small" style="margin-left:12px">Explain position</button>
    `;
    chatPanel.appendChild(tog);
    const incGames = document.getElementById('chat-inc-games');
    const incPrinciples = document.getElementById('chat-inc-principles');
    if(incGames){ incGames.addEventListener('change', ()=>{ CHAT_INCLUDE_GAMES = incGames.checked; }); }
    if(incPrinciples){ incPrinciples.addEventListener('change', ()=>{ CHAT_INCLUDE_PRINCIPLES = incPrinciples.checked; }); }
    const expBtn = document.getElementById('chat-explain-position');
    if(expBtn){ expBtn.addEventListener('click', ()=>{
      const summary = buildPositionSummaryTemplate();
      showPositionSummaryModal(summary, ()=>{
        // First push the structured summary into the chat context
        pushChat('user', summary);
        try{ CHAT.history.push({ role:'user', content: summary }); }catch(_){/* ignore */}
        // Then send the prompt asking to explain the position
        const input = document.getElementById('chat-input');
        if(input){ input.value = 'Explain the position'; }
        sendChat();
      });
    }); }
  }
  toggle.addEventListener('click', ()=>{
    const collapsed = ans.classList.toggle('collapsed');
    toggle.textContent = collapsed ? 'Show more' : 'Show less';
  });

  // click handling for annotated explanation and game text
  function handleTextClick(e){
    const t = e.target;
    if(t && t.classList.contains('gt-move')){
      const ply = parseInt(t.dataset.ply, 10);
      if(!Number.isNaN(ply)){
        currentPly = ply;
        renderPly();
      }
    }
  }
  ans.addEventListener('click', handleTextClick);

  // optional toggle for PGN tags via query ?pgnTags=0
  const pgnTags = URL_PARAMS.get('pgnTags');
  if(pgnTags === '0'){ window.__SHOW_PGN_TAGS__ = false; }

  // FEN panel wiring removed
});

// Global FEN helpers as a fallback to ensure the toggle always works
// Tactics panel wiring
  const tacticsBtn = document.getElementById('show-tactics');
  const tacticsList = document.getElementById('tactics-list');
  const tacticsCountEl = document.getElementById('tactics-count');
  const tacticsPanel = document.getElementById('tactics-panel');
  // Candidate Moves panel wiring
  const candidatesBtn = document.getElementById('show-candidates');
  const candidatesList = document.getElementById('candidate-list');
  const candidatesPanel = document.getElementById('candidate-panel');
  const candidatesCountEl = document.getElementById('candidates-count');
  // Minimal helper: derive tactics state from engine details (any id containing 'tactic'), then overlay highlights
  function tacticsStateFromPly(data){
    try{
      // Only count multi-move engine variations (line length >=2). Backend already filters strength/length.
      const vars = Array.isArray(data.variations) ? data.variations : [];
      const multi = vars.filter(v=>Array.isArray(v.line) && v.line.length>=2 && Array.isArray(v.fens) && v.fens.length>=2);
      const count = multi.length;
      return { count, enabled: count > 0 };
    }catch(_){ return { count: 0, enabled: false }; }
  }
  // Custom FEN panel removed
  function setBtnEnabled(btn, enabled){
    if(!btn) return;
    btn.disabled = !enabled;
    btn.classList.toggle('disabled', !enabled);
    btn.style.opacity = enabled ? '' : '0.5';
    btn.style.pointerEvents = enabled ? '' : 'none';
    btn.setAttribute('aria-disabled', String(!enabled));
  }
  async function showTactics(){
    // Prefer engine multi-move variations for tactics playback; fallback to overlay highlights
    try{
      if(!currentGameId){ setBtnEnabled(tacticsBtn, false); return; }
      const data = await fetchJSON(`${API_BASE}/ply`, { method:'POST', body: JSON.stringify({game_id: currentGameId, ply: currentPly}) });
      if(!data){ setBtnEnabled(tacticsBtn, false); return; }
      // Find a multi-move engine variation (>=2 moves)
      const vars = Array.isArray(data.variations) ? data.variations : [];
      const playable = vars.find(v=>Array.isArray(v.line) && v.line.length>=2 && Array.isArray(v.fens) && v.fens.length>=2);
      if(playable){
        startTacticsPlayback(playable);
        return;
      }
      // No multi-move line available: show a friendly message and do nothing.
      if(tacticsList){ tacticsList.innerHTML = '<li class="small">No multi-move tactics here.</li>'; }
    }catch(e){ if(tacticsList){ tacticsList.innerHTML = `<li class="error">Failed: ${escapeHTML(e.message||String(e))}</li>`; } }
  }
  if(tacticsBtn){ setBtnEnabled(tacticsBtn, false); tacticsBtn.addEventListener('click', showTactics); }
  async function updateTacticsCount(){
    try{
      if(!tacticsCountEl) return;
      // Use /ply overlays only in game mode
      const data = await fetchJSON(`${API_BASE}/ply`, { method:'POST', body: JSON.stringify({game_id: currentGameId, ply: currentPly}) });
      const ts = data ? tacticsStateFromPly(data) : { count: 0, enabled: false };
      tacticsCountEl.textContent = `(${ts.count})`;
      setBtnEnabled(tacticsBtn, ts.enabled);
    }catch(_){ if(tacticsCountEl){ tacticsCountEl.textContent = '(0)'; } setBtnEnabled(tacticsBtn, false); }
  }

  async function updateCandidatesState(){
    try{
      if(!candidatesBtn) return;
      if(!currentGameId){ setBtnEnabled(candidatesBtn, false); return; }
      const data = await fetchJSON(`${API_BASE}/candidates`, { method:'POST', body: JSON.stringify({game_id: currentGameId, ply: currentPly}) });
      const items = Array.isArray(data.candidates) ? data.candidates : [];
      if(candidatesCountEl){ candidatesCountEl.textContent = `(${items.length})`; }
      setBtnEnabled(candidatesBtn, items.length > 0);
    }catch(_){ if(candidatesBtn){ setBtnEnabled(candidatesBtn, false); } }
  }

  // Candidate Moves
  async function showCandidates(){
    try{
      if(!currentGameId){ return; }
      const data = await fetchJSON(`${API_BASE}/candidates`, { method:'POST', body: JSON.stringify({game_id: currentGameId, ply: currentPly}) });
      const items = Array.isArray(data.candidates) ? data.candidates : [];
      // Determine side to move to format SAN for black with '.. '
      let isBlackToMove = false;
      try{
        const fen = (window.__LAST_FEN__ || '').trim();
        const stm = (fen.split(' ')[1]||'').toLowerCase();
        isBlackToMove = (stm === 'b');
      }catch(_){/* ignore */}
      if(candidatesList){
        candidatesList.innerHTML = '';
        if(!items.length){ candidatesList.innerHTML = '<li class="small">No candidates</li>'; return; }
        items.forEach((v, idx)=>{
          const li = document.createElement('li');
          const cp = typeof v.score_cp === 'number' ? v.score_cp : null;
          const sign = (cp!==null && cp>0) ? '+' : '';
          const first = v.first_san || (v.line && v.line[0]) || '';
          const evalText = (cp!==null) ? `${sign}${cp} cp` : (v.label || '');
          const firstDisp = isBlackToMove ? (`.. ${first}`) : first;
          li.innerHTML = `<span class="var-move">${escapeHTML(firstDisp)}</span> <span class="small" style="opacity:.85">— ${escapeHTML(evalText)}</span>`;
          li.title = Array.isArray(v.line) ? v.line.join(' ') : '';
          li.addEventListener('mouseenter', ()=>{
            if(v.first_from && v.first_to){
              drawArrow(v.first_from, v.first_to, '#60a5fa');
              const svg = document.getElementById('board-overlay');
              const lines = svg ? svg.querySelectorAll('line') : null;
              const lastLine = lines && lines.length ? lines[lines.length-1] : null;
              if(lastLine){ lastLine.setAttribute('stroke-dasharray','2,2'); }
            }
          });
          li.addEventListener('mouseleave', ()=>{ clearOverlay(); });
          li.addEventListener('click', ()=>{ startCandidatePlayback(v); });
          candidatesList.appendChild(li);
        });
      }
    }catch(e){ if(candidatesList){ candidatesList.innerHTML = `<li class=\"error\">Failed: ${escapeHTML(e.message||String(e))}</li>`; } }
  }
  if(candidatesBtn){ setBtnEnabled(candidatesBtn, false); candidatesBtn.addEventListener('click', showCandidates); }

  function startCandidatePlayback(variation){
    try{
      const fens = Array.isArray(variation.fens) ? variation.fens.filter(Boolean) : [];
      if(fens.length < 1) return;
      // Use explicit panel if present, else fall back to the Candidates tab panel
      let panel = document.getElementById('candidate-panel');
      if(!panel){ panel = document.getElementById('tab-moves-candidates'); }
      const controlsId = 'candidate-controls';
      let controls = document.getElementById(controlsId);
      if(!controls){
        controls = document.createElement('div');
        controls.id = controlsId;
        controls.className = 'small';
        controls.style.marginTop = '6px';
        controls.innerHTML = `
          <button id="candidate-prev" class="small">Prev</button>
          <button id="candidate-next" class="small">Next</button>
          <button id="candidate-exit" class="small" style="margin-left:8px">Exit Candidates</button>
        `;
        panel.appendChild(controls);
      }
      const state = {
        active: true,
        index: 0,
        fens,
        restorePly: currentPly,
        restoreGame: currentGameId
      };
      window.__CANDIDATE_MODE__ = state;
      function setCandidateActive(idx){
        try{
          document.querySelectorAll('#candidate-list .cand-move').forEach(el=>{
            el.classList.remove('active');
          });
          const el = document.querySelector(`#candidate-list .cand-move[data-idx="${idx}"]`);
          if(el){ el.classList.add('active'); }
        }catch(_){/* ignore */}
      }
      async function renderIndex(){
        const idx = window.__CANDIDATE_MODE__.index;
        const fen = window.__CANDIDATE_MODE__.fens[idx];
        clearOverlay();
        renderFenToGrid(fen);
        // Update FEN string display
        try{
          const fenText = document.getElementById('fen-text');
          if(fenText){ fenText.textContent = `FEN: ${fen}`; }
        }catch(_){/* ignore */}
        // Refresh principle badges for this candidate position using /tactics overlays
        try{
          const resp = await fetchJSON(`${API_BASE}/tactics`, { method:'POST', body: JSON.stringify({ fen }) });
          const ov = (resp && resp.overlays) ? resp.overlays : { arrows: [], highlights: [] };
          window.__LAST_OVERLAYS__ = ov;
          const tags = Array.from(new Set((ov.highlights||[]).map(h=>h && h.principle).filter(Boolean)));
          renderPrinciples(tags, [], ov);
          // Draw overlay arrows for this position
          if(ov && Array.isArray(ov.arrows)){
            try{ console.log('[CAND] arrows:', ov.arrows.length, 'sample:', JSON.stringify(ov.arrows.slice(0,3))); }catch(_){/* ignore */}
            for(const a of ov.arrows){
              const color = (a && a.color) ? a.color.toLowerCase() : '';
              const ep = getArrowEndpoints(a);
              if(color === '#ff1a1a'){
                const adj = retargetAttackArrow({from:ep.from, to:ep.to}, fen);
                if(adj) drawArrow(adj.from, adj.to, a.color || '#ff1a1a');
              }else{
                if(ep && ep.from && ep.to) drawArrow(ep.from, ep.to, a.color || '#eab308');
              }
            }
          }
          // Highlight endpoints and any provided highlight squares
          try{
            const sqSet = new Set();
            if(ov && Array.isArray(ov.arrows)){
              for(const a of ov.arrows){ const ep = getArrowEndpoints(a); if(ep && ep.from) sqSet.add(ep.from); if(ep && ep.to) sqSet.add(ep.to); }
            }
            if(ov && Array.isArray(ov.highlights)){
              for(const h of ov.highlights){ if(h && h.square) sqSet.add(h.square); }
            }
            const impacted = Array.from(sqSet).map(sq=>({square: sq, principle: 'AttackedPieces'}));
            highlightSquares(impacted);
          }catch(_){/* ignore */}
        }catch(_){/* ignore */}
        if(idx>0){
          const prev = window.__CANDIDATE_MODE__.fens[idx-1];
          const move = inferMoveFromFen(prev, fen);
          if(move && move.from && move.to){ drawArrow(move.from, move.to, '#60a5fa'); }
        }
        setCandidateActive(idx);
      }
      renderIndex();
      const prevBtn = document.getElementById('candidate-prev');
      const nextBtn = document.getElementById('candidate-next');
      const exitBtn = document.getElementById('candidate-exit');
      if(prevBtn){ prevBtn.onclick = async ()=>{ if(window.__CANDIDATE_MODE__ && window.__CANDIDATE_MODE__.index>0){ window.__CANDIDATE_MODE__.index--; await renderIndex(); } }; }
      if(nextBtn){ nextBtn.onclick = async ()=>{ if(window.__CANDIDATE_MODE__ && window.__CANDIDATE_MODE__.index < window.__CANDIDATE_MODE__.fens.length-1){ window.__CANDIDATE_MODE__.index++; await renderIndex(); } }; }
      if(exitBtn){ exitBtn.onclick = async ()=>{
        try{
          window.__CANDIDATE_MODE__ = null;
          const c = document.getElementById(controlsId); if(c) c.remove();
          clearOverlay();
          await renderPly();
        }catch(_){/* ignore */}
      }; }
      // Also list the SAN line for reference, including centipawn eval if available
      if(candidatesList){
        candidatesList.innerHTML = '';
        if(typeof variation.score_cp === 'number'){
          const liEval = document.createElement('li');
          const cp = variation.score_cp;
          const sign = cp>0?'+':'';
          liEval.textContent = `Eval: ${sign}${cp} cp`;
          candidatesList.appendChild(liEval);
        }
        const line = Array.isArray(variation.line) ? variation.line : [];
        const baseMoveNo = Math.floor((typeof currentPly==='number' ? currentPly : 0) / 2) + 1;
        const whiteToMove = (typeof currentPly==='number' ? (currentPly % 2 === 0) : true);
        if(whiteToMove){
          for(let i=0;i<line.length;i+=2){
            const moveNo = baseMoveNo + Math.floor(i/2);
            const w = line[i] || '';
            const b = line[i+1] || '';
            const li = document.createElement('li');
            const wSpan = document.createElement('span');
            wSpan.className = 'move cand-move';
            wSpan.dataset.idx = String(i);
            wSpan.textContent = `${moveNo}. ${w}`;
            li.appendChild(wSpan);
            if(b){
              li.appendChild(document.createTextNode(' '));
              const bSpan = document.createElement('span');
              bSpan.className = 'move cand-move';
              bSpan.dataset.idx = String(i+1);
              bSpan.textContent = `${moveNo}... ${b}`;
              li.appendChild(bSpan);
            }
            candidatesList.appendChild(li);
          }
        }else{
          // First ply is black's move
          if(line.length>=1){
            const li0 = document.createElement('li');
            const bSpan0 = document.createElement('span');
            bSpan0.className = 'move cand-move';
            bSpan0.dataset.idx = '0';
            bSpan0.textContent = `${baseMoveNo}... ${line[0] || ''}`;
            li0.appendChild(bSpan0);
            candidatesList.appendChild(li0);
          }
          for(let i=1;i<line.length;i+=2){
            const moveNo = baseMoveNo + Math.floor((i+1)/2);
            const w = line[i] || '';
            const b = line[i+1] || '';
            const li = document.createElement('li');
            const wSpan = document.createElement('span');
            wSpan.className = 'move cand-move';
            wSpan.dataset.idx = String(i);
            wSpan.textContent = `${moveNo}. ${w}`;
            li.appendChild(wSpan);
            if(b){
              li.appendChild(document.createTextNode(' '));
              const bSpan = document.createElement('span');
              bSpan.className = 'move cand-move';
              bSpan.dataset.idx = String(i+1);
              bSpan.textContent = `${moveNo}... ${b}`;
              li.appendChild(bSpan);
            }
            candidatesList.appendChild(li);
          }
        }
      }
    }catch(_){/* noop */}
  }

  // Tactics playback mode
  function startTacticsPlayback(variation){
    try{
      const fens = Array.isArray(variation.fens) ? variation.fens.filter(Boolean) : [];
      if(fens.length < 2) return;
      // Use explicit panel if present, else fall back to the Tactics tab panel
      let panel = document.getElementById('tactics-panel');
      if(!panel){ panel = document.getElementById('tab-moves-tactics'); }
      const controlsId = 'tactics-controls';
      let controls = document.getElementById(controlsId);
      if(!controls){
        controls = document.createElement('div');
        controls.id = controlsId;
        controls.className = 'small';
        controls.style.marginTop = '6px';
        controls.innerHTML = `
          <button id="tactic-prev" class="small">Prev</button>
          <button id="tactic-next" class="small">Next</button>
          <button id="tactic-exit" class="small" style="margin-left:8px">Exit Tactics</button>
        `;
        panel.appendChild(controls);
      }
      const state = {
        active: true,
        index: 0,
        fens,
        restorePly: currentPly,
        restoreGame: currentGameId
      };
      window.__TACTICS_MODE__ = state;
      function setTacticsActive(idx){
        try{
          document.querySelectorAll('#tactics-list .tact-move').forEach(el=>{
            el.classList.remove('active');
          });
          const el = document.querySelector(`#tactics-list .tact-move[data-idx="${idx}"]`);
          if(el){ el.classList.add('active'); }
        }catch(_){/* ignore */}
      }
      async function renderIndex(){
        const idx = window.__TACTICS_MODE__.index;
        const fen = window.__TACTICS_MODE__.fens[idx];
        clearOverlay();
        renderFenToGrid(fen);
        // Update FEN string display
        try{
          const fenText = document.getElementById('fen-text');
          if(fenText){ fenText.textContent = `FEN: ${fen}`; }
        }catch(_){/* ignore */}
        // Refresh principle badges for this tactics position using /tactics overlays
        try{
          const resp = await fetchJSON(`${API_BASE}/tactics`, { method:'POST', body: JSON.stringify({ fen }) });
          const ov = (resp && resp.overlays) ? resp.overlays : { arrows: [], highlights: [] };
          window.__LAST_OVERLAYS__ = ov;
          const tags = Array.from(new Set((ov.highlights||[]).map(h=>h && h.principle).filter(Boolean)));
          renderPrinciples(tags, [], ov);
          // Draw overlay arrows for this position
          if(ov && Array.isArray(ov.arrows)){
            for(const a of ov.arrows){
              const color = (a && a.color) ? a.color.toLowerCase() : '';
              if(color === '#ff1a1a'){
                const adj = retargetAttackArrow(a, fen);
                if(adj) drawArrow(adj.from, adj.to, a.color);
              }else{
                drawArrow(a.from, a.to, a.color || '#eab308');
              }
            }
          }
          // Highlight endpoints and any provided highlight squares
          try{
            const sqSet = new Set();
            if(ov && Array.isArray(ov.arrows)){
              for(const a of ov.arrows){ if(a && a.from) sqSet.add(a.from); if(a && a.to) sqSet.add(a.to); }
            }
            if(ov && Array.isArray(ov.highlights)){
              for(const h of ov.highlights){ if(h && h.square) sqSet.add(h.square); }
            }
            const impacted = Array.from(sqSet).map(sq=>({square: sq, principle: 'AttackedPieces'}));
            highlightSquares(impacted);
          }catch(_){/* ignore */}
        }catch(_){/* ignore */}
        // draw arrow from previous to current where possible
        if(idx>0){
          const prev = window.__TACTICS_MODE__.fens[idx-1];
          const move = inferMoveFromFen(prev, fen);
          if(move && move.from && move.to){ drawArrow(move.from, move.to, '#eab308'); }
        }
        setTacticsActive(idx);
      }
      renderIndex();
      const prevBtn = document.getElementById('tactic-prev');
      const nextBtn = document.getElementById('tactic-next');
      const exitBtn = document.getElementById('tactic-exit');
      if(prevBtn){ prevBtn.onclick = async ()=>{ if(window.__TACTICS_MODE__ && window.__TACTICS_MODE__.index>0){ window.__TACTICS_MODE__.index--; await renderIndex(); } }; }
      if(nextBtn){ nextBtn.onclick = async ()=>{ if(window.__TACTICS_MODE__ && window.__TACTICS_MODE__.index < window.__TACTICS_MODE__.fens.length-1){ window.__TACTICS_MODE__.index++; await renderIndex(); } }; }
      if(exitBtn){ exitBtn.onclick = async ()=>{
        try{
          window.__TACTICS_MODE__ = null;
          // remove controls
          const c = document.getElementById(controlsId); if(c) c.remove();
          clearOverlay();
          // restore main game board
          await renderPly();
        }catch(_){/* ignore */}
      }; }
      // Also list the SAN line for reference, including centipawn eval if available
      if(tacticsList){
        tacticsList.innerHTML = '';
        if(typeof variation.score_cp === 'number'){
          const liEval = document.createElement('li');
          const cp = variation.score_cp;
          const sign = cp>0?'+':'';
          liEval.textContent = `Eval: ${sign}${cp} cp`;
          tacticsList.appendChild(liEval);
        }
        const line = Array.isArray(variation.line) ? variation.line : [];
        const baseMoveNo = Math.floor((typeof currentPly==='number' ? currentPly : 0) / 2) + 1;
        const whiteToMove = (typeof currentPly==='number' ? (currentPly % 2 === 0) : true);
        if(whiteToMove){
          for(let i=0;i<line.length;i+=2){
            const moveNo = baseMoveNo + Math.floor(i/2);
            const w = line[i] || '';
            const b = line[i+1] || '';
            const li = document.createElement('li');
            const wSpan = document.createElement('span');
            wSpan.className = 'move tact-move';
            wSpan.dataset.idx = String(i);
            wSpan.textContent = `${moveNo}. ${w}`;
            li.appendChild(wSpan);
            if(b){
              li.appendChild(document.createTextNode(' '));
              const bSpan = document.createElement('span');
              bSpan.className = 'move tact-move';
              bSpan.dataset.idx = String(i+1);
              bSpan.textContent = `${moveNo}... ${b}`;
              li.appendChild(bSpan);
            }
            tacticsList.appendChild(li);
          }
        }else{
          // First ply is black's move
          if(line.length>=1){
            const li0 = document.createElement('li');
            const bSpan0 = document.createElement('span');
            bSpan0.className = 'move tact-move';
            bSpan0.dataset.idx = '0';
            bSpan0.textContent = `${baseMoveNo}... ${line[0] || ''}`;
            li0.appendChild(bSpan0);
            tacticsList.appendChild(li0);
          }
          for(let i=1;i<line.length;i+=2){
            const moveNo = baseMoveNo + Math.floor((i+1)/2);
            const w = line[i] || '';
            const b = line[i+1] || '';
            const li = document.createElement('li');
            const wSpan = document.createElement('span');
            wSpan.className = 'move tact-move';
            wSpan.dataset.idx = String(i);
            wSpan.textContent = `${moveNo}. ${w}`;
            li.appendChild(wSpan);
            if(b){
              li.appendChild(document.createTextNode(' '));
              const bSpan = document.createElement('span');
              bSpan.className = 'move tact-move';
              bSpan.dataset.idx = String(i+1);
              bSpan.textContent = `${moveNo}... ${b}`;
              li.appendChild(bSpan);
            }
            tacticsList.appendChild(li);
          }
        }
      }
    }catch(_){/* noop */}
  }

document.getElementById('prev').onclick = async ()=>{
  if(currentPly>0){ currentPly--; await renderPly(); }
}

document.getElementById('next').onclick = async ()=>{
  currentPly++;
  await renderPly();
}

document.getElementById('ply').addEventListener('change', async (e)=>{
  currentPly = parseInt(e.target.value||'0',10) || 0; await renderPly();
});

document.getElementById('pgn').addEventListener('click', async (e)=>{
  const t = e.target;
  if(t && t.classList.contains('move')){
    const ply = parseInt(t.dataset.ply, 10);
    if(!Number.isNaN(ply)){
      currentPly = ply;
      await renderPly();
    }
  }
});

document.getElementById('export-pgn').onclick = ()=>{
  if(!currentGameId || !currentMoves.length) return;
  const text = movesToPGN(currentMoves);
  const safeId = String(currentGameId).replace(/[^a-z0-9-_]+/gi,'_');
  download(`${safeId}.pgn`, text);
}

// Removed Explain widget (#ask/#query); explanation now tied to game selection only.

// toggle overlays: global flag, helper function, and guard rendering overlays in renderPly based on the flag
// Enable overlays to show red borders for impacted squares.
window.__SHOW_OVERLAYS__ = true;
function toggleOverlays(){
  window.__SHOW_OVERLAYS__ = !window.__SHOW_OVERLAYS__;
  const btn = document.getElementById('toggle-overlays');
  if(btn){ btn.textContent = window.__SHOW_OVERLAYS__ ? 'Overlays: On' : 'Overlays: Off'; }
  console.log('[OVERLAYS] toggled:', window.__SHOW_OVERLAYS__);
  clearOverlay();
  document.querySelectorAll('.principle-border,.principle-border-space').forEach(el=>{
    el.classList.remove('principle-border','principle-border-space');
  });
  // Only re-render overlays if turning ON
  if(window.__SHOW_OVERLAYS__ && typeof renderPly === 'function') renderPly();
}

document.getElementById('toggle-overlays').onclick = toggleOverlays;

window.__SHOW_VARIATIONS__ = false;
function toggleVariations(){
  window.__SHOW_VARIATIONS__ = !window.__SHOW_VARIATIONS__;
  const btn = document.getElementById('toggle-variations');
  const panel = document.getElementById('variations-panel');
  if(btn){ btn.textContent = window.__SHOW_VARIATIONS__ ? 'Variations: On' : 'Variations: Off'; }
  if(panel){ panel.classList.toggle('hidden', !window.__SHOW_VARIATIONS__); }
}
function renderVariations(list){
  const panel = document.getElementById('variations-panel');
  const ul = document.getElementById('variations-list');
  if(!panel || !ul) return;
  ul.innerHTML = '';
  if(!window.__SHOW_VARIATIONS__){ panel.classList.add('hidden'); return; }
  if(!list || !list.length){ panel.classList.remove('hidden'); ul.innerHTML = '<li class="small">No variations</li>'; return; }
  for(const v of list){
    const li = document.createElement('li');
    const first = v.first_san || (v.line && v.line[0]) || '';
    const spanMove = document.createElement('span');
    spanMove.className = 'var-move';
    spanMove.textContent = first;
    const spanLabel = document.createElement('span');
    spanLabel.className = 'var-label';
    spanLabel.textContent = v.label ? ` — ${v.label}` : '';
    li.appendChild(spanMove);
    li.appendChild(spanLabel);
    // Hover preview: dashed arrow for first move
    li.addEventListener('mouseenter', ()=>{
      if(v.first_from && v.first_to){
        drawArrow(v.first_from, v.first_to, '#f59e0b');
        const svg = document.getElementById('board-overlay');
        const last = svg && svg.lastElementChild;
        if(last && last.tagName.toLowerCase()==='line'){
          last.setAttribute('stroke-dasharray','2,2');
        }
      }
    });
    li.addEventListener('mouseleave', ()=>{ clearOverlay(); });
    ul.appendChild(li);
  }
  panel.classList.remove('hidden');
}
