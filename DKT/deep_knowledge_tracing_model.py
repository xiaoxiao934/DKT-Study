import torch.nn as nn


class DeepKnowledgeTracing(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size, num_skills, nlayers, dropout=0.6, tie_weights=False):
        super(DeepKnowledgeTracing, self).__init__()

        # 选择网络类型, 生成RNN的隐层
        if rnn_type == 'LSTM':
            # input_size: 输入数据的特征维数, hidden_size: LSTM中隐层的维度, nlayers: 网络层数
            self.rnn = nn.LSTM(input_size, hidden_size, nlayers, batch_first=True, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, nlayers, batch_first=True, dropout=dropout)
        elif rnn_type == 'RNN_TANH':
            self.rnn = nn.RNN(input_size, hidden_size, nlayers, nonlinearity='tanh', dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, nlayers, nonlinearity='relu', dropout=dropout)

        #  创建一个全连接层，作为解码器（decoder），输入层的维度为hidden_size，输出层的维度为num_skills。
        #  这一层将隐层（self.rnn）的输出映射到目标技能数量（num_skills）的维度。
        self.decoder = nn.Linear(hidden_size, num_skills)

        # 初始化权重及其他
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = nlayers  # 循环神经网络（RNN）的层数.通过设置nlayers，可以控制RNN中层数的数量

    def init_weights(self):
        initrange = 0.05  # 初始化权重的范围
        # 隐层到输出层的网络的权重
        self.decoder.bias.data.zero_()  # 将解码器（self.decoder）的偏置（bias）初始化为零
        # 将解码器（self.decoder）的权重（weight）从均匀分布中抽样初始化，范围为[-initrange, initrange]。
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # 前向计算, 网络结构是：input --> hidden(self.rnn) --> decoder(输出层)
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html?highlight=rnn#torch.nn.RNN
    # 根据官网,torch.nn.RNN接收的参数input形状是[时间步数, 批量大小, 特征维数], hidden: 旧的隐藏层的状态
    def forward(self, input, hidden):
        # output: 隐藏层在各个时间步上计算并输出的隐藏状态, 形状是[时间步数, 批量大小, 隐层维数]
        output, hidden = self.rnn(input, hidden)
        # decoded: 形状是[时间步数, 批量大小, num_skills]
        # 将output经过一个view操作重塑为二维形状，并通过解码器（self.decoder）进行线性映射，得到decoded，它是对目标技能数量的预测
        decoded = self.decoder(output.contiguous().view(output.size(0) * output.size(1), output.size(2)))
        # 返回decoded和最后一个时间步的隐藏状态hidden
        return decoded, hidden
        # 初始化隐藏状态
    def init_hidden(self, bsz):
        # 获取模型的第一个参数的权重
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
