from pydantic import BaseModel


class Params(BaseModel):
    def save(self, path):
        with open(path, "w") as f:
            f.write(self.model_dump_json())
