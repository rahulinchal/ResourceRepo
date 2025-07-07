# Python Programming Problems & Solutions

A comprehensive guide to common Python interview questions with detailed solutions, explanations, and pro tips.

---

## 1. Swap Keys and Values in Dictionary

### Problem Statement
Given a dictionary, swap the keys and values to create a new dictionary where original values become keys and original keys become values.

### Solution
```python
dict1 = {'a': 1, 'b': 2, 'c': 3}
print("Original keys:", dict1.keys())
print("Original values:", dict1.values())

# Dictionary comprehension to swap keys and values
dict2 = {value: key for key, value in dict1.items()}
print("Swapped dictionary:", dict2)
print("New keys:", dict2.keys())
print("New values:", dict2.values())
```

### Expected Output
```
Original keys: dict_keys(['a', 'b', 'c'])
Original values: dict_values([1, 2, 3])
Swapped dictionary: {1: 'a', 2: 'b', 3: 'c'}
New keys: dict_keys([1, 2, 3])
New values: dict_values(['a', 'b', 'c'])
```

### Explanation
- `dict1.items()` returns key-value pairs as tuples
- Dictionary comprehension `{value: key for key, value in dict1.items()}` iterates through each pair and swaps positions
- This creates a new dictionary with swapped mappings

### 💡 Pro Tips
- **Edge Case**: Be careful with duplicate values in original dictionary - they'll overwrite each other as keys
- **Interview Tip**: Mention that this approach assumes all values are hashable (can be dictionary keys)
- **Alternative**: You can also use `dict(zip(dict1.values(), dict1.keys()))`

---

## 2. Find Missing Element in Counting Numbers

### Problem Statement
Given a list of consecutive numbers with one missing element, find the missing number.

### Solution
```python
a = [1, 2, 3, 5]
l = len(a) + 1  # Expected length including missing number

# Create complete range
b = [x for x in range(1, l + 1)]
print("Complete range:", b)

# Find missing element using set difference
res = list(set(b) - set(a))
print("Missing element:", res)
```

### Expected Output
```
Complete range: [1, 2, 3, 4, 5]
Missing element: [4]
```

### Explanation
- Calculate expected length: `len(a) + 1` (original length plus missing element)
- Generate complete sequence from 1 to expected length
- Use set difference to find missing elements
- Convert result back to list

### 💡 Pro Tips
- **Mathematical Approach**: `missing = sum(range(1, n+2)) - sum(a)` is more efficient
- **Interview Tip**: Discuss time complexity - Set approach: O(n), Mathematical: O(n) but single pass
- **Edge Case**: Handle empty lists or lists with multiple missing numbers

---

## 3. Difference Between `==` and `is` in Python

### Problem Statement
Explain and demonstrate the difference between equality (`==`) and identity (`is`) operators in Python.

### Solution
```python
list1 = [1, 2, 3]
list2 = [1, 2, 3]

print("list1 == list2:", list1 == list2)  # True - same content
print("list1 is list2:", list1 is list2)  # False - different objects
print("id(list1):", id(list1))
print("id(list2):", id(list2))
```

### Expected Output
```
list1 == list2: True
list1 is list2: False
id(list1): 140234567890123  # (example memory address)
id(list2): 140234567890456  # (different memory address)
```

### Explanation
- **`==` (Equality)**: Compares the **values/content** of objects
- **`is` (Identity)**: Compares **memory addresses** - checks if both variables reference the same object
- Two lists with identical content are equal (`==`) but not identical (`is`)

### 💡 Pro Tips
- **Common Gotcha**: Small integers (-5 to 256) and short strings are cached by Python, so `is` might return `True`
- **Interview Tip**: Always use `is` with `None`: `if variable is None:`
- **Best Practice**: Use `==` for value comparison, `is` only for singleton objects like `None`, `True`, `False`

---

## 4. Count Unique Values from List

### Problem Statement
Given a list with duplicate elements, count the number of unique values.

### Solution
```python
a = [1, 2, 2, 3, 3, 3, 3, 4, 4, 5]

# Convert to set to get unique elements, then count
b = set(a)
unique_count = len(b)
print("Unique elements:", b)
print("Count of unique values:", unique_count)
```

### Expected Output
```
Unique elements: {1, 2, 3, 4, 5}
Count of unique values: 5
```

### Explanation
- `set(a)` removes all duplicate elements, keeping only unique values
- `len()` counts the number of elements in the set
- Sets automatically handle deduplication

### 💡 Pro Tips
- **Alternative Methods**: `len(Counter(a))` or `len(pd.Series(a).unique())` for pandas
- **Interview Tip**: Mention that sets are unordered, so original order is lost
- **Performance**: Set conversion is O(n) average case, very efficient for large lists

---

## 5. Find Most Frequent Element

### Problem Statement
Given a list of numbers, find the element that appears most frequently.

### Solution
```python
num = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5]

from collections import Counter

# Count frequencies
counter = Counter(num)
print("Frequency count:", counter)

# Get most common element
most_frequent = counter.most_common()[0][0]
print("Most frequent element:", most_frequent)
```

### Expected Output
```
Frequency count: Counter({5: 4, 3: 3, 2: 2, 4: 2, 1: 1})
Most frequent element: 5
```

### Explanation
- `Counter` creates a dictionary-like object with elements as keys and their counts as values
- `most_common()` returns a list of tuples sorted by frequency in descending order
- `[0][0]` gets the first tuple's first element (the most frequent item)

### 💡 Pro Tips
- **Get Top N**: Use `counter.most_common(3)` to get top 3 most frequent elements
- **Interview Tip**: Discuss handling ties - `most_common()` returns arbitrary order for equal frequencies
- **Manual Approach**: You can also use `max(counter, key=counter.get)` for single most frequent

---

## 6. Find Duplicate Elements from List

### Problem Statement
Given a list, find all elements that appear more than once.

### Solution
```python
a = [1, 1, 2, 2, 2, 3, 4, 4]

from collections import Counter

# Count frequencies
counter = Counter(a)
print("All frequencies:", counter.items())

# Filter elements with frequency > 1
duplicates = [item for item, freq in counter.items() if freq > 1]
print("Duplicate elements:", duplicates)
```

### Expected Output
```
All frequencies: dict_items([(1, 2), (2, 3), (3, 1), (4, 2)])
Duplicate elements: [1, 2, 4]
```

### Explanation
- `Counter(a)` counts frequency of each element
- List comprehension filters items where frequency > 1
- Returns list of all elements that appear multiple times

### 💡 Pro Tips
- **Alternative**: Use set to track seen and duplicate elements in single pass
- **Interview Tip**: Discuss space-time tradeoffs - Counter uses O(n) space but is very readable
- **Preserve Order**: Use `collections.OrderedDict` if order matters

---

## 🎯 General Interview Tips

### Time Complexity Discussion
- Always analyze and discuss the time and space complexity of your solutions
- Consider edge cases and alternative approaches
- Think about scalability for large datasets

### Code Quality
- Use descriptive variable names
- Add comments for complex logic
- Follow Python naming conventions (PEP 8)

### Problem-Solving Approach
1. **Understand**: Clarify the problem requirements
2. **Plan**: Think through the approach before coding
3. **Code**: Write clean, readable solution
4. **Test**: Consider edge cases and validate output
5. **Optimize**: Discuss improvements and alternatives

### Common Python Interview Topics
- Data structures (lists, dicts, sets, tuples)
- List comprehensions and generator expressions
- Built-in functions (map, filter, reduce, zip)
- Collections module (Counter, defaultdict, deque)
- Exception handling
- Object-oriented programming concepts
