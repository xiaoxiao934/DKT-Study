import csv
import random

# 该函数用于加载数据文件并对数据进行预处理，生成一个数据集。数据集中的每个样本由题目个数、题目序列和答对情况组成。同时，函数还返回了最长题目序列的长度和知识点的最大编号
def load_data(fileName):
    rows = []  # 创建一个空列表，用于存储从数据文件中读取的行
    max_skill_num = 0  # 用于记录知识点（题目）的最大数量
    max_num_problems = 0  # 记录最长题目序列的长度
    with open(fileName, "r") as csvfile:  # 打开数据文件，使用 csv 模块来读取文件内容
        reader = csv.reader(csvfile, delimiter=',')  # 创建一个 CSV 读取器，指定逗号为分隔符,将每行数据存入rows列表中
        for row in reader:
            # row可能是学生id或做题序列或答对01序列
            rows.append(row)
    print("filename: " + fileName + "the number of rows is " + str(len(rows)))  # 打印文件名和读取的行数

    index = 0  # 初始化index遍历rows列表
    tuple_rows = []  # 创建空列表 tuple_rows，用于存储处理后的数据集
    while index < len(rows)-1:
        problems_num = int(rows[index][0])  # 将当前行的第一个元素（题目个数）转换为整数，并赋值给变量 problems_num
        rows[index + 1].remove('')
        tmp_max_skill = max(map(int, rows[index+1]))  # 将下一行的每个元素转换为整数，并取得其中的最大值，赋值给 tmp_max_skill。它表示题目序列中的最大知识点编号
        if tmp_max_skill > max_skill_num:
            max_skill_num = tmp_max_skill  # 更新 max_skill_num 的值,记录最大的知识点编号
        if problems_num <= 2:  # 如果题目个数小于等于 2，跳过当前行和下两行，直接将索引 index 增加 3
            index += 3
        else:
            if problems_num > max_num_problems:
                max_num_problems = problems_num  # 更新 max_num_problems 的值为题目个数，以记录最长题目序列的长度
            # 创建一个元组 tup，其中包含当前行、下一行和下两行的内容。这个元组表示一个样本，包含题目个数、题目序列和答对情况
            tup = (rows[index], rows[index+1], rows[index+2])
            # tup:[题目个数, 题目序列, 答对情况]
            tuple_rows.append(tup)  # 将 tup 添加到 tuple_rows 列表中，将其作为一个样本加入数据集
            index += 3  # 增加索引 index 的值，以便处理下一个样本
    # shuffle the tuple
    random.shuffle(tuple_rows)  # 使用 random 模块的 shuffle() 函数对 tuple_rows 列表中的样本进行随机打乱，以增加数据集的随机性
    # tuple_rows的每一行是tup:[[题目个数], [题目序列], [答对情况]], max_num_problems最长题目序列, max_skill_num是知识点(题目)个数
    return tuple_rows, max_num_problems, max_skill_num+1
    # 返回处理后的数据集 tuple_rows、最长题目序列的长度 max_num_problems，以及知识点（题目）的最大编号 max_skill_num+1
