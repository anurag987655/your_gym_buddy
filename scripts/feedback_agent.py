import json
import os

from groq import Groq


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
    def __init__(self, api_key=None):
        self.model = "llama-3.3-70b-versatile"
        self.client = Groq(api_key=api_key) if api_key else None
        self.system_prompt = (
            "You are a professional yoga and fitness instructor.\n"
            "Rewrite the provided deterministic coaching cue in a concise, supportive tone.\n"
            "Do not change the meaning, target body part, safety priority, or movement phase.\n"
            "Return one line, max 15 words."
        )

    def get_feedback(self, state_packet):
        deterministic_cue = state_packet.get("deterministic_cue") or state_packet.get("local_hint")
        if not deterministic_cue:
            return "Keep your form controlled and safe."

        if not self.client:
            return deterministic_cue

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Deterministic cue: {deterministic_cue}\n"
                            f"State JSON: {json.dumps(state_packet, ensure_ascii=True)}"
                        ),
                    },
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=40,
            )
            text = chat_completion.choices[0].message.content.strip()
            return text if text else deterministic_cue
        except Exception:
            return deterministic_cue


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    load_env_file(os.path.join(project_root, ".env"))
    api_key = os.getenv("GROQ_API_KEY")
    agent = FeedbackAgent(api_key)
    print(
        agent.get_feedback(
            {
                "pose": "squat_bad_back",
                "knee_angle": 95,
                "deterministic_cue": "Lift chest and brace core to keep a neutral spine.",
            }
        )
    )
