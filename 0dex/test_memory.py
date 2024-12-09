from main import Memory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import time
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def setup_memory():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        collection_name="memory_test",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        streaming=True,
        openai_api_key=OPENAI_API_KEY
    )
    return Memory(vectorstore=vectorstore, embeddings=embeddings, llm=llm)

def simulate_conversation(memory):
    test_prompts = [
        "Hi, I'm Sarah! Let's talk about space exploration.",
        "What can you tell me about NASA's current missions?",
        "That's interesting! How about their Mars rovers?",
        "Let's switch topics. What do you know about coffee?",
        "I love cappuccinos. What's your favorite coffee drink?",
        "Actually, can we go back to talking about Mars rovers?",
        "What about SpaceX? Are they competing with NASA?",
        "Let's talk about something completely different. Do you know any good recipes?",
        "I'm interested in Italian cuisine.",
        "Actually, let's go back to space. What was that about the Mars rovers again?"
    ]
    
    for prompt in test_prompts:
        print("\n" + "="*50)
        print(f"\nUser: {prompt}")
        response = memory.process_prompt(prompt)
        print(f"\nAssistant: {response}")
        print(f"\nWorking Memory: {memory.working_memory}")
        time.sleep(2)  # Add delay between prompts for readability

def main():
    memory = setup_memory()
    print("Memory system initialized. Enter test mode? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        print("\nStarting automated test conversation...")
        simulate_conversation(memory)
        return
    
    print("Starting interactive mode. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break
                
            response = memory.process_prompt(user_input)
            print("\nAssistant:", response)
            print("\nWorking Memory:", memory.working_memory)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 