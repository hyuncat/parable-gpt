import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat

class ParableGPT:
    def __init__(self, llm="llama3.1:8b", embedder="all-MiniLM-L6-v2"):
        self.LLM_MODEL = llm
        self.EMBEDDING_MODEL = SentenceTransformer(embedder)
        self.DB_PATH = "./corpus/chroma_db"
        self.client = chromadb.PersistentClient(self.DB_PATH)

        # tradition-specific configuration variables
        self.TRADITION_CONFIG = {
            "Christianity": {
                "collection": "bible",
                "style": (
                    "Use a reverent, Biblical tone. "
                    "Prefer simple, concrete imagery. "
                    "Do not imitate any specific modern author."
                ),
                "source_label": "Bible passages",
                "ref_formatter": lambda m: f'{m["book"]} {m["chapter"]}:{m["start_verse"]}-{m["end_verse"]}',
            },
            "Buddhism": {
                "collection": "dhammapada",
                "style": (
                    "Use a calm, concise, contemplative tone similar to the provided sources. "
                    "Avoid sermonizing; let the lesson emerge naturally."
                ),
                "source_label": "Dhammapada passages",
                "ref_formatter": lambda m: f'Dhammapada {m["chapter"]} vv.{m["start_verse"]}-{m["end_verse"]}',
            },
            "Islam": {
                "collection": "quran",
                "style": (
                    "Use a poetic, rhythmic tone similar to the Quranic style. "
                    "Incorporate vivid imagery and metaphors."
                ),
                "source_label": "Quranic passages",
                "ref_formatter": lambda m: f'Surah {m["surah"]} vv.{m["start_verse"]}-{m["end_verse"]}',
            },
            "Taoism": {
                "collection": "tao_te_ching",
                "style": (
                    "Use a simple, paradoxical, and poetic tone similar to the Tao Te Ching. "
                    "Emphasize naturalness and spontaneity."
                ),
                "source_label": "Tao Te Ching passages",
                "ref_formatter": lambda m: f'Chapter {m["chapter"]}',
            }
        }

    def retrieve(self, tradition, topic, k: int=6) -> tuple[list[str], list[dict]]:
        """Retrieves k relevant verses from the collection based on the topic
        also returns relevant metadata such as book, chapter, verse number 
        (varies by tradition)

        Args:
            tradition (str): the tradition/collection to query from
            topic (str): the topic to base the retrieval on
            k (int, optional): number of verses to retrieve. Defaults to 6.
        Returns:
            tuple[list[str], list[dict]]: retrieved verses and their metadata
        """
        theme_embeddings = self.EMBEDDING_MODEL.encode([topic], normalize_embeddings=True).tolist()
        col = self.client.get_collection(name=self.TRADITION_CONFIG[tradition]["collection"])
        results = col.query(
            query_embeddings=theme_embeddings,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        verses = results['documents'][0]
        metadatas = results['metadatas'][0]
        return verses, metadatas


    def generate(self, tradition: str, topic: str, length: int=150, info: str=None) -> tuple[str, str]:
        """Generates a parable in the style of the specified tradition,
        focusing on the given topic, length, and following any additional 
        instructions from the prompt.
        
        Args:
            tradition (str): the tradition/style to emulate
            topic (str): the topic/theme of the parable
            length (int, optional): desired word count of the parable. Defaults to 150
            info (str, optional): additional instructions for the parable. Defaults to None.
        Returns:
            tuple[str, str]: generated parable and the sources used
        """
        # retrive relevant verses
        verses, metadatas = self.retrieve(tradition, topic)
        cfg = self.TRADITION_CONFIG[tradition]

        # format sources
        sources = []
        for v, m in zip(verses, metadatas):
            ref = cfg["ref_formatter"](m)
            sources.append(f"{ref}\n{v}")
        sources_text = "\n\n".join(sources)

        # SYSTEM
        system = (
            f"You are ParableGPT, generating a parable in the style of {tradition}. "
            f"{cfg['style']} "
            "Write an original parable inspired by the provided sources. "
            "Do not quote long passages verbatim; paraphrase ideas instead. "
            f"{f'Target length: about {length} words (±15%). ' if length!='' else ''}"
            "Begin with EXACTLY: 'Title: [insert_parable_title_here]'. "
            "End with EXACTLY: 'Moral: [insert_moral_here]'. Be concise."
        )
        # USER: topic + constraints
        user = (
            f"Topic: {topic}\n\n"
            "User constraints (follow these carefully):\n"
            f"{(info or "").strip()}\n\n"
            f"Relevant {cfg['source_label']} "
            "(Imitate this tone and writing style as exactly as possible)"
            # "while following the Title: [insert_parable_title_here] and Moral: [insert_moral_here] format clearly.):"
            f"\n\n{sources_text}\n\n"
            "Now write the parable."
        )
        resp = chat(
            model=self.LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        parable = resp["message"]["content"]
        return parable, sources
    
    def run(self):
        """
        Keep prompting user for input and generating parables until they quit.
        """
        print("Welcome to ParableGPT!\n")
        while True:
            while True:
                tradition_index = input("Select a tradition below or 'q' to quit: \n" \
                                        " (0) Christianity\n (1) Buddhism\n" \
                                        " (2) Islam\n (3) Taoism\n" \
                                        "Your choice: ").strip()
                if tradition_index.lower() == 'q':
                    return
                try:
                    idx = int(tradition_index)
                    if 0 <= idx < len(list(self.TRADITION_CONFIG.keys())):
                        break
                except ValueError:
                    pass
                print(f"Please enter a number 0-{len(list(self.TRADITION_CONFIG.keys()))-1}, or 'q'.")

            tradition = list(self.TRADITION_CONFIG.keys())[int(tradition_index)]
            topic = input("Enter topic for the parable: ").strip()
            length = input("Enter desired word count (enter to skip): ").strip()
            info = input("Enter any additional instructions for the parable (enter to skip): ").strip()
            parable, sources = self.generate(tradition, topic, length, info)
            print("\n---\n")
            print(parable)
            print("\n---\n")

if __name__ == "__main__":
    parable_gpt = ParableGPT()
    parable_gpt.run()