from groq import Groq
import os

class FeedbackAgent:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.system_prompt = """
        You are a professional yoga and fitness instructor. 
        Your goal is to provide actionable, encouraging feedback based on the user's current posture state.
        Keep corrections extremely concise (under 15 words).
        Focus on immediate improvement and positive reinforcement.
        """

    def get_feedback(self, state_packet):
        """
        state_packet: dict e.g., {"pose": "squat_bad_back", "knee_angle": 85, "back_angle": 30}
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
                        "content": f"User State: {state_packet}",
                    }
                ],
                model=self.model,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Keep it up! (Agent error: {str(e)})"

if __name__ == "__main__":
    API_KEY = os.getenv("GROQ_API_KEY")
    if not API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Export it before running.")
    agent = FeedbackAgent(API_KEY)
    print(agent.get_feedback({"pose": "squat_bad_back", "knee_angle": 95}))
