import requests
import json

class LLM:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "phi3:3.8b-mini-4k-instruct-q4_K_M"

    def stream(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }

        response = requests.post(self.url, json=payload, stream=True)

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))

            if "response" in data:
                yield data["response"]

# local unit testing code
if __name__ == "__main__":
    time_log = []
    llm = LLM()
    prompt = "Tell me a joke on AI progress."

    import time
    for i in range(15):
        start_time = time.time()
        result = ""
        for response in llm.stream(prompt):
            result+=response
        end_time = time.time()
        time_log.append(end_time - start_time)
        print(f"Result {i+1}: {result}")

    print(f"Minimum time: {min(time_log)}")
    print(f"Average time: {sum(time_log) / len(time_log)}")
    print(f"Maximum time: {max(time_log)}")