class RuntimeState:
    def __init__(self):
        self.history_limit = 20
        self.max_summaries = 15
        self.max_vectors_per_session = 200

runtime = RuntimeState()