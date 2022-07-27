from builtins import str
from typing import List


class Solution:
    def get_longest_zero(self, nums: List[int]) -> int:
        n = 0
        count_n = []
        if len(nums) == nums.count(0):
            return len(nums)
        for num in nums:
            if num == 0:
                n += 1
            else:
                count_n.append(n)
                n = 0
        return max(count_n)



s = Solution()
strs = []
# strs = ["flower", "flow", "flight"]
print(s.get_longest_zero(strs))
