OPENAI_API_KEY="..."

if __name__ == '__main__':
    from langchain import OpenAI

    ai = OpenAI() #get codex instance

    ai = OpenAI(
        model_name="code-davinci-002", # Codex model
        temperature=0.7,               # Higher values mean more random
        max_tokens=100,               # Maximum output length
        frequency_penalty=0,          # Penalize repetitive text
        presence_penalty=0            # Penalize off-prompt wandering
    )

    import langchain

    ai = langchain.OpenAI()


    def generate_post(title, author):
        prompt = f"Generate a blog post on '{title}' by {author}:"

        return ai.generate(prompt)


    print(generate_post("Using LangChain", "Shriniwas Iyengar"))