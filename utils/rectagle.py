class Rectangle:
    x = 0
    y = 0
    w = 0
    h = 0

    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


    def top(self):      return self.y
    def bottom(self):   return self.y + self.h
    def left(self):     return self.x
    def right(self):    return self.x + self.w

    def tl(self):       return self.x, self.y
    def br(self):       return self.right(), self.bottom()
    def area(self):     return self.w * self.h

    def __width(self, left, right):     return right - left
    def __height(self, top, bottom):    return bottom - top

    def union(self, other, margin=0):
        top = max(self.top(), other.top()) - margin
        bottom = min(self.bottom(), other.bottom()) + margin
        left = max(self.left(), other.left()) - margin
        right = min(self.right(), other.right()) + margin

        width = max(self.__width(left, right), 0)
        height = max(self.__height(top, bottom), 0)

        if width == 0: left = (self.left() + other.left()) / 2
        if height == 0: top = (self.top() + other.top()) / 2

        return Rectangle(left, top, width, height)

    def is_balanced(self):
        ratio = self.w / self.h
        return ratio

    def __add__(self, other):
        top = min(self.top(), other.top())
        bottom = max(self.bottom(), other.bottom())
        left = min(self.left(), other.left())
        right = max(self.right(), other.right())

        width = self.__width(left, right)
        height = self.__height(top, bottom)

        return Rectangle(left, top, width, height)

    def __mul__(self, other):
        return self.union(other)

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.w == other.w and self.h == other.h:
            return True
        else:
            return False

    def str(self):
        return str(str(self.x) + ' ' + str(self.y) + ' ' + str(self.w) + ' '  + str(self.h))

    def print_(self):
        print(str(self.x) + ' ' + str(self.y) + ' ' + str(self.w) + ' '  + str(self.h))
