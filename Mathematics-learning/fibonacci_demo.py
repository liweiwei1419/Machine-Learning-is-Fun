class Fibonacci(object):
    ''' 返回一个 fibonacci 数列'''

    def __init__(self):
        self.fList = [0, 1]  # 设置初始的列表
        self.main()

    def main(self):
        listLen = input("请输入的 fibonacci 数列的长度：")
        while len((self.fList)) < int(listLen):
            self.fList.append(self.fList[-1] + self.fList[-2])
        print("得到的 Fibonacci 数列为：\n %s" % self.fList)

    def checkLen(self,lenth):
        lenList = map(str,range(3,51))

        for item in lenList:
            print(item)

if __name__ == '__main__':
    f = Fibonacci()
    f.checkLen(10)

