class ShelveReader():
    def __init__(self, db, name=None):
        self.db = db
        self.pos = -1
        self.stop = len(db)
        self.name = name

    def __iter__(self):
        return self

    def __getitem__(self, i):
        return self.db[str(i)]

    def decrease_iter(self):
        self.pos -= 1

    def __next__(self):
        self.pos += 1
        if self.pos >= self.stop:
            raise StopIteration

        if self.name:
            print('{}: {}; stop: {}'.format(self.name, self.pos, self.stop))
        return self.db[str(self.pos)]