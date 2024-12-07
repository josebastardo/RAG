from decouple import config
import os

class Config:
    OPENAI_API_KEY = config('openai_api_key', default=None)
    OPENAI_BASE_URL = config('openai_base_url', default="https://openrouter.ai/api/v1")
    MODEL = config('model', default="https://api.deepseek.com")
    HF_API_KEY = config('pc_api_key', default=None)
    
    @staticmethod
    def validate_config():
        required_vars = {
            'OPENAI_API_KEY': Config.OPENAI_API_KEY,
            'OPENAI_BASE_URL': Config.OPENAI_BASE_URL,

        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
