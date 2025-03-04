To run frontend,
>cd frontend

To install Angular
>npm install -g @angular/cli
>npm install

Build and run Angular frontend
>ng build
>ng serve

To run backend,
>cd backend

To create specific environment for python
>python3 -m venv .venv
>source .venv/bin/activate
>pip3 install -r requirements.txt

Add OPENAI key to use OPENAI API
>export OPENAI_API_KEY=<OPENAI_API_KEY>

Edit backend.env and set the CORS_ORIGINS parameter to the actual IP address or pattern

Run backend RAG program
>python ER_chat_service.py --env-file backend.env
