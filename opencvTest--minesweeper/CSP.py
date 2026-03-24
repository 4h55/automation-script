import numpy as np
from dataclasses import dataclass,field
from collections import deque
import copy
@dataclass()
class Block:
    position:(int,int)
    range: list[int] = field(default_factory=lambda: [0, 1])
@dataclass
class Constraint:
    vars: list[Block]
    target: int
class GetVariables:
    def __init__(self,positions):
        self.positions = positions
        self.total_rows, self.total_cols = positions.shape  # 全局棋盘的总行数/总列数
        self.variables,self.constraints=self.get_variables_and_constraints()
        self.csp=CSP(self.variables)



    def get_neighbors(self,row,col):
        directions = np.array([
            [-1, -1], [-1, 0], [-1, 1],[0, -1],[0, 1],[1, -1],[1, 0], [1, 1]
        ])
        neighbors = np.array([row, col]) + directions
        valid_mask = (neighbors[:, 0] >= 0) & (neighbors[:, 0] <self.total_rows) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < self.total_cols)
        vaild_neighbors = neighbors[valid_mask]
        return vaild_neighbors

    def get_variables_and_constraints(self):
        variable_positions = set()
        constraints=[]
        num_mask = (self.positions >= 1) & (self.positions <= 4)
        nums = np.argwhere(num_mask)
        for row, col in nums:
            neighbors = self.get_neighbors(row, col)
            for nr, nc in neighbors:
                if self.positions[nr, nc] == -1:
                    variable_positions.add((int(nr), int(nc)))

        variables = []
        pos_to_var = {}
        for pos in variable_positions:
            block = Block(pos)
            variables.append(block)
            pos_to_var[pos] = block
        for num in nums:
            row, col = num
            target = self.positions[row,col]
            neighbors=self.get_neighbors(row,col)
            unopened_mask = (self.positions[neighbors[:, 0], neighbors[:, 1]] == -1)
            unopened_neighbors = neighbors[unopened_mask].tolist()
            var_list=[pos_to_var[(row,col)] for row,col in unopened_neighbors if (row,col) in pos_to_var]
            if len(var_list) > 0:
                constraints.append(Constraint(vars=var_list, target=target))
        return variables,constraints
class CSP:
    def __init__(self,variables):
        self.variables = variables
        self.edges=[]

    def add_edges(self,scope,number):
        if set(scope).issubset(set(self.variables)):
            self.edges.append({"scope":scope,"number":number})
        else:
            raise KeyError("Scope not in variables")

    def _deep_copy_vars(self):
        """深拷贝变量，返回 {位置: 新Block对象} 的字典"""
        new_blocks = {}
        for var in self.variables:
            # 直接用 position 做 key，不需要 old_to_new 字典
            new_blocks[var.position] = Block(var.position, var.range.copy())
        return new_blocks

    def _clone_constraints(self, constraints, new_blocks):
        """
        复制约束，并通过 position 从 new_blocks 中查找新的变量引用
        """
        new_constraints = []
        for c in constraints:
            # 关键修改：通过 var.position 去 new_blocks 里找新对象
            new_vars = [new_blocks[var.position] for var in c.vars]
            new_constraints.append(Constraint(new_vars, c.target))
        return new_constraints
    def check(self,constraint:Constraint,idx,value):
        target_value=constraint.target
        block_num = len(constraint.vars)
        valid_masks=[]

        for mask in range(0,1<<block_num):
            is_valid=True
            total=0

            for i in range(0,block_num):
                val=1 if mask & (1<<i) else 0
                if val not in constraint.vars[i].range:
                    is_valid=False
                    break
                total+=val
            if is_valid and total == target_value:
                valid_masks.append(mask)
        if not valid_masks:
            return False

        valid_mask = [mask for mask in valid_masks if ((mask & (1 << idx)) != 0) == value]
        if not valid_mask:
            return False
        return True
    def gac(self,constraints:list[Constraint]):
        queue=deque(constraints)
        while queue:
            constraint=queue.popleft()

            for var_idx,var in enumerate(constraint.vars):
                original_range=var.range.copy()
                for val in original_range:
                    result=self.check(constraint,var_idx,val)
                    if not result:
                        var.range.remove(val)
                        for c in constraints:
                            if c is not constraint and var in c.vars:
                                queue.append(c)
                if not var.range:
                    return False
        return True

    def get_safe_and_mine(self, constraints: list[Constraint], positions):
        safe_block = {}
        if not self.gac(constraints):
            return safe_block
        for var in self.variables:
            if len(var.range) == 1:
                val = var.range[0]
                if val == 0:
                    row, col = var.position
                    x1 = positions[0, col]
                    y1 = positions[row, 0]
                    safe_block[var.position] = (x1, y1)
                continue
            # 假设是雷，矛盾则安全
            new_blocks= self._deep_copy_vars()
            test_constraints = self._clone_constraints(constraints, new_blocks)

            test_var = new_blocks[var.position]
            test_var.range = [1]
            temp_csp = CSP(list(new_blocks.values()))
            is_possible_mine = temp_csp.gac(test_constraints)
            if not is_possible_mine:
                row, col = var.position
                x2 = positions[0, col]
                y2 = positions[row, 0]
                safe_block[var.position] = (x2, y2)
                var.range = [0]
                continue
            #假设安全
            new_blocks = self._deep_copy_vars()
            test_constraints = self._clone_constraints(constraints, new_blocks)

            test_var = new_blocks[var.position]
            test_var.range = [0]
            temp_csp = CSP(list(new_blocks.values()))
            is_possible_mine = temp_csp.gac(test_constraints)
            if not is_possible_mine:
                var.range = [1]


        return safe_block

    '''
    def solve(self,constraints:list[Constraint]):
        if not self.gac(constraints):
            return False
        result=[var for var in self.variables if len(var.range)>1]
        if not result:
            return {v.position: v.range[0] for v in self.variables}
        var = min(result, key=lambda v: len(v.range))
        save_range={var.position:var.range.copy() for var in self.variables}
        for val in var.range:
            var.range=[val]
            if self.gac(constraints):
                result=self.solve(constraints)
                if result:
                    return result

            for v in self.variables:
                v.range=save_range[v.position]
        return None
    '''


'''
def main():
    positions = np.array([
        [0, 36, 118, 186, 205, 268, 337, 360, 416, 481],
        [30, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [86, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [142, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [199, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [255, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [311, -1, -1, -1, -1, -1, 2, 1, 2, -1],
        [367, -1, -1, -1, 2, 1, 1, 0, 0, 0],
        [423, -1, -1, -1, 1, 0, 0, 0, 0, 0],
        [480, -1, -1, -1, 1, 0, 0, 0, 0, 0]
    ])
    # 运行求解
    print("正在构建CSP...")
    builder = GetVariables(positions)
    builder.get_variables_and_constraints()
    print(f"变量数: {len(builder.variables)}, 约束数: {len(builder.constraints)}")

    



if __name__ == "__main__":
    main()
'''
'''
1.变量数=数字值，全是雷
2.变量数为n和n-1的超边，如果重叠的点的数量是n-1，可以确定一个变量的值
3.
'''
'''
2. 约束传播

如果某个 v 没有支持，则从 x 的值域中删除 v，并将所有包含 x 的其他约束（除了 C）加入队列。
传播结束后，可能出现三种情况：
某个变量值域为空 → 当前分支无解，需回溯。
所有变量值域均为单值 → 得到一个完整解。
仍有变量值域为 {0,1} → 需要搜索。

3. 回溯搜索

选择一个未确定的变量,依次尝试该变量的每个可能值：
将当前状态的所有变量值域保存，回溯时恢复。
将该变量的值域设为单值。
将所有包含该变量的约束加入队列，再次运行约束传播。
如果传播后无矛盾，则递归进入下一层搜索。
如果传播导致矛盾，则回溯，恢复状态，尝试下一个值。
当所有变量都确定且满足所有约束时，记录一个解。如果要找所有解，则继续回溯探索其他分支；如果只需一个解，则可终止。
import itertools
all_combinations = itertools.combinations(cards, 3)
'''