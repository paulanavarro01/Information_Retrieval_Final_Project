import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class RAGGenerator:

    PROMPT_TEMPLATE = """
You are an expert e-commerce advisor. Your task is to pick the best product for the user.

Rules:
- Always consider color, gender, type (from user query).
- Prioritize products with lower price, higher discount, and higher rating.
- Keep explanations short and factual.
- Suggest one alternative only if relevant.
- Output exactly as:
  Best Product: [PID] [Title] - Why: [Reason]. 
  Alternative: [PID] [Title] - Why: [Reason]. (optional)
- If no product fits, output:
  "There are no good products that fit the request."

Products (PID | Title | Price | Discount | Rating | Brand | Category):
{retrieved_results}

User Query:
{user_query}


## Output Format:
- Best Product: ...
- Why: ...
- Alternative (optional): ...
"""

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 10) -> str:
        """
        Generate a RAG response based on retrieved results.
        """

        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."

        try:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                print("❌ GROQ_API_KEY not found in environment.")
                return DEFAULT_ANSWER

            client = Groq(api_key=api_key)
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            # Format top-N results with structured info
            formatted_results = []
            for res in retrieved_results[:top_N]:
                details = getattr(res, "product_details", {})
                details_str = ", ".join(f"{k}: {v}" for k, v in details.items()) if details else "No extra details"
                formatted_results.append(
                    f"- PID: {res.pid}, Title: {res.title}, Price: ₹{res.selling_price}, Discount: {res.discount}%, "
                    f"Rating: {res.average_rating}/5, Brand: {getattr(res, 'brand', 'N/A')}, "
                    f"Category: {getattr(res, 'category', 'N/A')}, Details: {details_str}"
                )

            prompt_text = self.PROMPT_TEMPLATE.format(
                retrieved_results="\n".join(formatted_results),
                user_query=user_query
            )

            # Request completion from the model
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"❌ Error during RAG generation: {e}")
            return DEFAULT_ANSWER
