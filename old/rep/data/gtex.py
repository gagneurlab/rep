from data.data import AbstractData

class GTEx(AbstractData):

    def __init__(self):
        super()

    @property
    def annobj(self):
        return super().annobj

    @property
    def varanno(self):
        return super().varanno

    @property
    def obsanno(self):
        return super().obsanno

