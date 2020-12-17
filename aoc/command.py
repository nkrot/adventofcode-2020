
class Command(object):

    # Example of what subclasses can do

    # @classmethod
    # def occupy(cls, seat):
    #     obj = cls(seat, 'occupy')
    #     return obj

    # @classmethod
    # def leave(cls, seat):
    #     obj = cls(seat, 'leave')
    #     return obj

    def __init__(self, receiver: object, meth: str):
        self.receiver = receiver
        self.method = meth
        self.debug = False

    def execute(self):
        """Execute the command"""
        assert hasattr(self.receiver, self.method), \
            f"Object {self.receiver} does not have attribute {self.method}"
        if self.debug:
            print(f"Before command: {repr(self.receiver)}")
        ret = getattr(self.receiver, self.method)()
        if self.debug:
            print(f"After command : {repr(self.receiver)}")
        return ret

    def __call__(self):
        return self.execute()
