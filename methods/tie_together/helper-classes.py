# Script containing helper classes for all-mem.py

# %%
class Tree:
    def __init__(self, data = None):
        self.data = data
    
    def GrowTree(self, depth):
        if depth==1:
            self.data = list([0,1])
            return self

        elif depth > 1:
            curr_level = 1
            self.data = list([0,1])

            curr_level = 2
            while curr_level <= depth:
                # Sweep through all leaves at the current level
                list_curr_level = list(np.repeat(np.nan, repeats=2**curr_level))
                for i in range(0, len(self.data)):
                    left_leaf = np.append(np.array(self.data[i]), 0)
                    right_leaf = np.append(np.array(self.data[i]), 1)
                    list_curr_level[2*i] = list(left_leaf)
                    list_curr_level[2*i + 1] = list(right_leaf)
                    #print(list_curr_level)
                    
                # Go one level below
                self.data = list_curr_level
                curr_level += 1
            return self

        else:
            return 0