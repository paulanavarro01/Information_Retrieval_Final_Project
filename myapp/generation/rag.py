import os
from dataclasses import dataclass
from typing import List, Any, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv() #load .env file if present

# dataclass for product representation
@dataclass  #automatically generates init, repr, eq, etc.
class Product:
    pid: str
    title: str
    brand: str
    category: str
    selling_price: Optional[float]
    average_rating: Optional[float]
    out_of_stock: bool
    product_url: str
    description: str


class RAGGenerator:
    # Constant for no good products message
    NO_GOOD_PRODUCTS = "There are no good products that fit the request."

    # Prompt template for RAG generation
    PROMPT_TEMPLATE = """
You are an expert e-commerce advisor. Your task is to pick the best product for the user. Use ONLY the retrieved products below, do NOT invent or assume any products or attributes. Base your recommendation strictly on the attributes provided.

Follow these rules strictly:
- Always consider color, gender, type (from user query).
- Prioritize products with lower price, higher discount, and higher rating.
- Keep explanations short and factual.
- Suggest one alternative only if relevant.
- Output exactly as: Best Product: [PID] [Title] - Why: [Reason].
  Alternative: [PID] [Title] - Why: [Reason]. (optional)
- If no product fits, output:
  "There are no good products that fit the request."

Products (PID | Title | Price | Discount | Rating | Brand | Category):
{retrieved_results}

User Query: {user_query}

## Output Format:
- Best Product: ...
- Why: ...

Alternative (optional): ...
"""
    # Initialize RAGGenerator with model and API key from environment variables
    def _init_(self,model_name_env: str = "GROQ_MODEL",api_key_env: str = "GROQ_API_KEY",default_model: str = "llama-3.1-8b-instant",) -> None:
        # Environment variable names and default model
        self.model_env_name = model_name_env
        self.api_key_env_name = api_key_env
        self.default_model = default_model
        api_key = os.environ.get(self.api_key_env_name) #retrieve API key from env
        # Initialize Groq client if API key is present
        if not api_key:
            self.client = None
        else:
            self.client = Groq(api_key=api_key)

        # Retrieve model name from environment or use default
        self.model_name = os.environ.get(self.model_env_name, self.default_model)

   

    @staticmethod
    def parse_float(value: Any) -> Optional[float]:
        """
        Converts values to float, returns None if conversion fails.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def format_product(self, res: Any) -> str:
        """
        Converts a product result into a formatted text block (string) for the prompt.
        """
        # Extract product attributes with defaults
        pid = getattr(res, "pid", None) or getattr(res, "id", "<no-pid>")
        title = getattr(res, "title", "") or ""
        brand = getattr(res, "brand", "") or ""
        category = getattr(res, "category", "") or ""
        price = getattr(res, "selling_price", None)
        rating = getattr(res, "average_rating", None)
        out_of_stock = getattr(res, "out_of_stock", False)
        url = getattr(res, "product_url", "") or ""
        description = getattr(res, "description", "") or ""

        # Create a snippet of the description (first 200 chars)
        description_snippet = (description[:200] + "..." if len(description) > 200 else description)

        #Format product text for the LLM prompt
        return (
            f"PID: {pid}\n"
            f"Title: {title}\n"
            f"Brand: {brand}\n"
            f"Category: {category}\n"
            f"Price: ₹{price}\n"
            f"Rating: {rating}/5\n"
            f"InStock: {not out_of_stock}\n"
            f"URL: {url}\n"
            f"Description: {description_snippet}\n"
            "----"
        )

    @staticmethod
    def format_summary_text(raw: str) -> str:
        """
        Extarcts only Best Product and Alternative lines from the LLM output.
        """
        if not raw:
            return ""

        best_line = None
        alt_line = None

        #parse line one by one
        for line in raw.splitlines():
            stripped = line.strip()
            lower = stripped.lower()

            #detect best product line
            if lower.startswith("best product:"):
                best_line = stripped

            #detect alternative line
            elif lower.startswith("alternative"):
                if lower in {"alternative: none", "alternative : none"}:
                    continue
                alt_line = stripped
       
        #if best product not found, return full raw text
        if not best_line:
            return raw.strip()
       
        #return formatted best product and alternative lines if any
        if alt_line:
            return f"{best_line}\n{alt_line}".strip()
        return best_line.strip()


    def filter_products_for_rag(self, retrieved_results: List[Any]) -> List[Any]:
        """
        Filters out products that are out of stock, lack description, or have low ratings.
        """
        filtered: List[Any] = []

        for res in retrieved_results:
            # Skip out-of-stock products
            if getattr(res, "out_of_stock", False):
                continue
            
            # Skip products without a valid description
            desc = getattr(res, "description", None)
            if not desc or str(desc).strip() == "":
                continue
            
            # Skip products with rating below 3.0
            rating_val = getattr(res, "average_rating", None)
            rating = self.parse_float(rating_val)
            if rating is not None and rating < 3.0:
                continue

            filtered.append(res)

        # If no products passed the filters, return the original list
        return filtered if filtered else retrieved_results

    def rank_products_for_rag(self, results: List[Any]) -> List[Any]:
        """
        Ranks products based on a scoring function that considers rating and price.
        """
        def score(r: Any) -> float:
            #extract rating and price values
            rating_val = getattr(r, "average_rating", None)
            price_val = getattr(r, "selling_price", None)

            rating = self.parse_float(rating_val) or 0.0
            price = self.parse_float(price_val)
            if price is None: #assign high default price if missing
                price = 1e6

            #score = rating - price_penalty
            price_penalty = price / (price + 1000.0)
            return rating - price_penalty

        try:
            return sorted(results, key=score, reverse=True)
        except Exception:
            return results


    def build_prompt(self, user_query: str, candidates: List[Any]) -> str:
        """
        Build the RAG prompt with formatted product candidates.
        """
        formatted_results = "\n".join(
            [self.format_product(r) for r in candidates]
        )
        return self.PROMPT_TEMPLATE.format(
            retrieved_results=formatted_results,
            user_query=user_query,
        )

    def generate_response(self,user_query: str,retrieved_results: List[Any], top_N: int = 10,temperature: float = 0.0, max_tokens: int = 512,) -> str:
        """
        Generate a RAG response based on retrieved results.
        """
        #message if API key is missing or error occurs
        DEFAULT_ANSWER = ("RAG is not available. Check your credentials (.env file) or account limits.")

        #ensure it is a list
        if not isinstance(retrieved_results, list):
            retrieved_results = list(retrieved_results or [])

        #if no products retrieved, return no good products message
        if not retrieved_results:
            return self.NO_GOOD_PRODUCTS

        #if client not initialized, return default answer
        if self.client is None:
            return DEFAULT_ANSWER

        try:
            #pre-filter and rank products
            filtered = self.filter_products_for_rag(retrieved_results)
            reranked = self.rank_products_for_rag(filtered)
            
            #choose top N candidates
            prompt_candidates = reranked[:top_N]

            if not prompt_candidates:
                return self.NO_GOOD_PRODUCTS

            #final prompt construction
            prompt = self.build_prompt(user_query, prompt_candidates)

            #prepare messages for chat completion
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert e-commerce advisor. "
                        "Follow the instructions strictly and only use provided products."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            #call Groq chat completion API
            chat_completion = self.client.chat.completions.create(messages=messages,model=self.model_name,temperature=temperature,max_completion_tokens=max_tokens,)

            #extract and format the generated response
            raw_generation = chat_completion.choices[0].message.content or ""
            formatted = self.format_summary_text(raw_generation)
            return formatted or DEFAULT_ANSWER

        except Exception as e:
            print(f"❌ Error during RAG generation: {e}")
            return DEFAULT_ANSWER