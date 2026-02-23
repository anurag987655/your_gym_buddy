from groq import Groq
import os
import json


def load_env_file(env_path):
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key.startswith("export "):
                key = key.replace("export ", "", 1).strip()
            if key and key not in os.environ:
                os.environ[key] = value


class FeedbackAgent:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.system_prompt = (
            "You are a professional yoga and fitness instructor.\n"
            "Return one concise coaching cue (max 15 words).\n"
            "Use only the pose label and metrics in the state packet.\n"
            "Prioritize safety-critical corrections first.\n"
            "Never invent metrics that are not provided.\n"
            "If state_packet.local_hint exists, refine it rather than changing topic."
        )

    def get_feedback(self, state_packet):
        """
        state_packet: dict e.g., {"pose": "squat_bad_back", "knee_angle": 85, "local_hint": "..."}
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"User State JSON: {json.dumps(state_packet, ensure_ascii=True)}",
                    }
                ],
                model=self.model,
                temperature=0.2,
                max_tokens=40,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            local_hint = state_packet.get("local_hint")
            if local_hint:
                return local_hint
            return f"Keep it up! (Agent error: {str(e)})"

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    load_env_file(os.path.join(project_root, ".env"))
    API_KEY = os.getenv("GROQ_API_KEY")
    if not API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Add it in .env or export it before running.")
    agent = FeedbackAgent(API_KEY)
    print(agent.get_feedback({"pose": "squat_bad_back", "knee_angle": 95}))
