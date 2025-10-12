const API_BASE = window.API_BASE || 'http://localhost:8000';

async function fetchJSON(url, opts={}){
  const res = await fetch(url, {headers:{'Content-Type':'application/json'}, ...opts});
  if(!res.ok){
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.json();
}

async function loadGames(){
  const games = await fetchJSON(`${API_BASE}/games`);
  const ul = document.getElementById('games');
  ul.innerHTML = '';
  games.forEach((g, i)=>{
    const li = document.createElement('li');
    li.textContent = `[${i+1}] ${g.id} (len=${g.length})`;
    li.onclick = ()=>selectGame(g.id);
    ul.appendChild(li);
  })
}

let currentGameId = null;
let currentPly = 0;

async function selectGame(id){
  currentGameId = id;
  currentPly = 0;
  document.getElementById('game-title').textContent = id;
  await renderPly();
}

async function renderPly(){
  if(!currentGameId) return;
  const data = await fetchJSON(`${API_BASE}/ply`, {method:'POST', body: JSON.stringify({game_id: currentGameId, ply: currentPly})});
  document.getElementById('board').textContent = data.board;
  document.getElementById('san').textContent = data.san.join(' ');
  document.getElementById('ply').value = currentPly;
}

document.getElementById('prev').onclick = async ()=>{
  if(currentPly>0){ currentPly--; await renderPly(); }
}

document.getElementById('next').onclick = async ()=>{
  currentPly++; await renderPly();
}

document.getElementById('ply').addEventListener('change', async (e)=>{
  currentPly = parseInt(e.target.value||'0',10) || 0; await renderPly();
});

document.getElementById('ask').onclick = async ()=>{
  const q = document.getElementById('query').value.trim();
  if(!q) return;
  const data = await fetchJSON(`${API_BASE}/explain`, {method:'POST', body: JSON.stringify({query: q})});
  document.getElementById('answer').textContent = data.answer;
}

loadGames().catch(err=>{
  document.getElementById('games').innerHTML = `<li style="color:#f87171">Failed to load games: ${err.message}</li>`;
});
