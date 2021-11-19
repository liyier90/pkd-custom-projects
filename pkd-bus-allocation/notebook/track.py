class Track:
    def __init__(self, track_id, tracker_constructor):
        self.track_id = track_id
        self._tracker_constructor = tracker_constructor
        self._tracker = None

        self.bbox = None
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0

    def update(self, frame, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        self.bbox = tuple(bbox)
        self._tracker = self._tracker_constructor()
        return self._tracker.init(frame, self.bbox)

    def get_state(self, frame):
        return self._tracker.update(frame)

    def predict(self, frame):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state(frame)
