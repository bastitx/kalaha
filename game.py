class Game():
    def __init__(self):
        self.fields = []
        for i in range(14): # 6 and 13 are stores
            self.fields += [4]
        self.active_player = 0

    def reset(self):
        for i in range(14):
            self.fields[i] = 4
        self.active_player = 0
        return self.get_state()

    def step(self, action):
        choice_pos = convert_to_position(int(action), self.active_player)
        prev_rew = self.get_state()[6]
        reward = 0
        if choice_pos is not None:
            switch = self.move(choice_pos)
            done = self.game_over()
            if done:
                player_fields = self.get_state()
                player_fields[6] += sum(player_fields[0:6])
                player_fields[13] += sum(player_fields[7:13])
                for i in range(6)+range(7,13):
                    player_fields[i] = 0

            if done and self.get_state()[6] > self.get_state()[13]:
                reward = 100
            elif done:
                reward = -100
            else:
                reward = self.get_state()[6] - prev_rew - 0.2

            if switch:
                if self.active_player == 0:
                    self.active_player = 1
                else:
                    self.active_player = 0

        return self.get_state(), reward, done, self.active_player

    def action_space(self):
        space = []
        for i in range(6):
            if self.get_state()[i] > 0:
                space += [i + 1]
        return space

    def get_state(self):
        if self.active_player == 0:
            return self.fields
        else:
            return self.fields[7:] + self.fields[:7]

    # def show(self):
    #     state = self.get_state()
    #     print('   ', end='')
    #     for i in range(12,6,-1):
    #         print('{0:2d} '.format(state[i]), end='')
    #     print('\n{0:2d}                   {1:2d}'.format(state[13], state[6]))
    #     print('   ', end='')
    #     for i in range(0,6):
    #         print('{0:2d} '.format(state[i]), end='')
    #     print('')


    def play(self):
        while not self.game_over():
            self.show()
            choice = input("It's player {0}'s turn. Select a field {1}: ".format(self.active_player, self.action_space()))
            self.step(choice)
        no_stones_0 = 0
        no_stones_1 = 0
        for i in range(0,6):
            no_stones_0 += self.fields[i]
        for i in range(7,13):
            no_stones_1 += self.fields[i]
        if no_stones_0 > no_stones_1:
            print('Player 0 won!')
        elif no_stones_0 < no_stones_1:
            print('Player 1 won!')
        else:
            print("It's a tie!")


    def move(self, pos):
        no_stones = self.fields[pos]
        if no_stones == 0:
            return False
        self.fields[pos] = 0
        i = pos
        while no_stones > 0:
            i += 1
            if self.active_player == 0 and i == 13:
                i = 0
            if self.active_player == 1 and i == 6:
                i = 7
            if i > 13:
                i = 0
            self.fields[i] += 1
            no_stones -= 1
        if self.active_player == 0 and i == 6:
            return False
        if self.active_player == 1 and i == 13:
            return False
        if self.fields[i] == 1 and self.fields[12-i] > 0:
            if self.active_player == 0 and i < 6:
                self.fields[6] += self.fields[12-i] + 1
                self.fields[12-i] = 0
                self.fields[i] = 0
            if self.active_player == 1 and 6 < i < 13:
                self.fields[13] += self.fields[12-i] + 1
                self.fields[12-i] = 0
                self.fields[i] = 0
        return True

    def game_over(self):
        if self.fields[6] > 29 or self.fields[13] > 29:
            return True
        go = True
        for i in range(6):
            if self.fields[i] != 0:
                go = False
                break
        if go:
            return True
        for i in range(7,13):
            if self.fields[i] != 0:
                return False
        return True

def convert_to_position(field, player):
    if 0 < field < 7 and 0 <= player <= 1:
        return field - 1 + player * 7
    else:
        return None

#game = Game()
#game.play()
