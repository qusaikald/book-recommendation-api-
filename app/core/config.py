from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_NAME: str = 'BAAI/bge-small-en-v1.5'
    CSV_PATH: str = 'books.csv'
    FAISS_INDEX_PATH: str = 'books.faiss'
    TOP_K: int = 5
    BATCH_SIZE: int = 32

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

settings = Settings()
