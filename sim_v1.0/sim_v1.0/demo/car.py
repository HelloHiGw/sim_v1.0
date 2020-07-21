class Car:
    def __init__(self, team=0, x=50, y=50, rotate=0, yaw=0, HP=2000, bullet=200):
        self.team   = team # 队伍
        self.x      = x # x坐标
        self.y      = y # y坐标
        self.rotate = rotate # 底盘方向角
        self.yaw    = yaw # 云台方向角
        self.HP     = HP # 生命值
        self.bullet = bullet # 子弹数   
