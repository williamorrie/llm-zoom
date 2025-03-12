from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

client = OpenAI()


def get_gpt_resp(question):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user",
        "content": f"{question}?"}]
    )
    return response


if __name__ == "__main__":
    get_gpt_resp("What time is love")