import google.generativeai as genai
import os

class FeedbackAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
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
        prompt = f"{self.system_prompt}
User State: {state_packet}
Instruction:"
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Keep it up! (Agent error: {str(e)})"

if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
    agent = FeedbackAgent(API_KEY)
    print(agent.get_feedback({"pose": "squat_bad_back", "knee_angle": 95}))
