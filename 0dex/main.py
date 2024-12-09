import uuid
from dataclasses import dataclass, field
from typing import Dict, List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os

load_dotenv()
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))


@dataclass
class Memory:
    vectorstore: Chroma
    embeddings: OpenAIEmbeddings
    llm: ChatOpenAI
    working_memory: dict = field(default_factory=lambda: {
        'user_name': None,
        'current_topic': None,
        'topic_summary': '',
        'context': {}
    })

    def compute_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.embeddings.embed_query(text1)
        emb2 = self.embeddings.embed_query(text2)
        return 1 - cosine(emb1, emb2)

    def check_context_switch(self, prompt: str) -> bool:
        if not self.working_memory['current_topic']:
            return False
            
        # Create context representation
        current_context = f"""
        Topic: {self.working_memory['current_topic']}
        Summary: {self.working_memory['topic_summary']}
        """
        
        system_prompt = f"""
        Analyze if this new prompt represents a context switch from the current conversation.
        
        Current Context:
        {current_context}
        
        New Prompt: {prompt}
        
        Respond with only "YES" or "NO" and a brief reason why.
        """
        
        response = self.llm.invoke([SystemMessage(content=system_prompt)]).content
        is_switch = response.upper().startswith("YES")
        
        if is_switch:
            print(f"\n[Context Switch Detected]: {response}")
            self.update_long_term_memory()
        
        return is_switch

    def update_long_term_memory(self):
        if not self.working_memory['topic_summary']:
            return
            
        print("\n[Storing to Long-term Memory]")
        memory_text = f"""
        Topic: {self.working_memory['current_topic']}
        User: {self.working_memory['user_name']}
        Summary: {self.working_memory['topic_summary']}
        Context: {self.working_memory['context']}
        """
        
        # Store in vectorstore
        self.vectorstore.add_texts([memory_text])
        print(f"Stored summary about: {self.working_memory['current_topic']}")

    def retrieve_relevant_context(self, prompt: str) -> str:
        # Get relevant documents from vectorstore
        docs = self.vectorstore.similarity_search(prompt, k=3)
        
        if not docs:
            return ""
        
        # Format the retrieved context
        retrieved_contexts = []
        for doc in docs:
            retrieved_contexts.append(doc.page_content)
        
        print("\n[Retrieved Memory]")
        print("\n".join(retrieved_contexts))
        
        return "\n".join(retrieved_contexts)

    def process_prompt(self, prompt: str) -> str:
        # Check for context switch
        if self.check_context_switch(prompt):
            self.update_long_term_memory()
            # Reset topic-related memory but keep user info
            user_name = self.working_memory['user_name']
            self.working_memory = {
                'user_name': user_name,
                'current_topic': None,
                'topic_summary': '',
                'context': {}
            }
        
        # Retrieve relevant context
        relevant_context = self.retrieve_relevant_context(prompt)
        
        system_prompt = f"""
        You are a conversational AI with memory. Process this prompt and maintain context.
        
        Current Memory State:
        - User Name: {self.working_memory.get('user_name')}
        - Current Topic: {self.working_memory.get('current_topic')}
        - Topic Summary: {self.working_memory.get('topic_summary')}
        
        Previous Relevant Context: {relevant_context}
        Current User Prompt: {prompt}
        
        Instructions:
        1. Use the previous context if the topic has been discussed before
        2. If the topic is new, provide a new topic name
        3. Update the topic summary to include:
           - What the user asked/said
           - How you responded
           - Key points discussed
           - Keep it concise but informative
        4. Provide a natural response that incorporates previous knowledge
        
        Example Summary Format:
        "User asked about NASA rockets. Discussed Saturn V and its role in Apollo missions. 
        Explained current SLS development. User showed interest in Mars missions."
        
        Respond in this format:
        MEMORY_UPDATES: (JSON with 'current_topic' and/or 'topic_summary' if needed)
        RESPONSE: (your response to the user)
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        result = self.llm.invoke(messages).content
        
        try:
            if "MEMORY_UPDATES:" in result and "RESPONSE:" in result:
                memory_updates_str = result.split("MEMORY_UPDATES:")[1].split("RESPONSE:")[0].strip()
                response = result.split("RESPONSE:")[1].strip()
                
                if memory_updates_str and memory_updates_str.lower() != "none":
                    import json
                    try:
                        updates = json.loads(memory_updates_str)
                        if updates != self.working_memory:  # Only print if there are actual changes
                            print("\n[Memory Updated]")
                            print(f"Topic: {updates.get('current_topic')}")
                            if 'topic_summary' in updates:
                                print(f"Summary: {updates.get('topic_summary')}")
                        self.working_memory.update(updates)
                    except json.JSONDecodeError:
                        print("\n[Failed to parse memory updates]")
                
                return response
        except Exception as e:
            print(f"\n[Error parsing response: {e}]")
        
        return result
