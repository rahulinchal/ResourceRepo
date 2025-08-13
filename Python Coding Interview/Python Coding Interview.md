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

## 7. Remove Outliers Using IQR Method

### Problem Statement
Given a dataset, remove outliers using the Interquartile Range (IQR) method.

### Solution
```python
a = [1, 11, 12, 13, 14, 100]

import numpy as np

# Calculate quartiles
q1 = np.percentile(a, 25)  # First quartile
q3 = np.percentile(a, 75)  # Third quartile

print(f"Q1: {q1}, Q3: {q3}")

# Calculate IQR and bounds
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr

print(f"IQR: {iqr}")
print(f"Bounds: [{lower_bound}, {upper_bound}]")

# Filter outliers
filtered_data = [x for x in a if lower_bound <= x <= upper_bound]
print("Data without outliers:", filtered_data)
```

### Expected Output
```
Q1: 11.5, Q3: 13.5
IQR: 2.0
Bounds: [-1.5, 16.5]
Data without outliers: [1, 11, 12, 13, 14]
```

### Explanation
- **IQR Method**: Standard statistical method for outlier detection
- **Q1 (25th percentile)**: 25% of data falls below this value
- **Q3 (75th percentile)**: 75% of data falls below this value
- **Outlier Definition**: Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

### 💡 Pro Tips
- **Alternative Methods**: Z-score method `abs(z) > 3`, Modified Z-score using median
- **Interview Tip**: Discuss when to use different outlier detection methods
- **Domain Knowledge**: Sometimes "outliers" are actually important data points

---

## 8. Count Word Frequencies in Text

### Problem Statement
Given a text string, count the frequency of each word (case-sensitive).

### Solution
```python
text = 'Python Programming python programming'

# Split text into words
words = text.split()
print("Words list:", words)

from collections import Counter

# Count word frequencies
word_counter = Counter(words)
print("Word frequencies:", word_counter)

# Get most common word
most_common_word = word_counter.most_common()[0][0]
print("Most frequent word:", most_common_word)
```

### Expected Output
```
Words list: ['Python', 'Programming', 'python', 'programming']
Word frequencies: Counter({'Python': 1, 'Programming': 1, 'python': 1, 'programming': 1})
Most frequent word: Python
```

### Explanation
- `split()` divides text into list of words using whitespace as delimiter
- `Counter` counts occurrences of each word
- Note: This is case-sensitive ('Python' ≠ 'python')

### 💡 Pro Tips
- **Case-Insensitive**: Use `text.lower().split()` for case-insensitive counting
- **Better Tokenization**: Use `re.findall(r'\b\w+\b', text.lower())` to handle punctuation
- **Interview Tip**: Discuss preprocessing steps - removing punctuation, handling contractions, stemming
- **Advanced**: Consider using NLTK or spaCy for production text processing

---

## 9. Check if String is Palindrome

### Problem Statement
Determine whether a given string reads the same forwards and backwards (palindrome check).

### Solution
```python
a = 'Maalayalam'
a = a.lower()  # Convert to lowercase for case-insensitive comparison

if a == a[::-1]:
    print("Yes")
else:
    print('No')
```

### Expected Output
```
Yes
```

### Explanation
- Convert string to lowercase using `lower()` for case-insensitive comparison
- `a[::-1]` creates a reversed version of the string using slice notation
- Compare original (lowercased) string with its reverse
- If they match, it's a palindrome

### 💡 Pro Tips
- **Two-Pointer Approach**: More memory efficient for very long strings
```python
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```
- **Interview Tip**: Ask if spaces and punctuation should be ignored
- **Advanced**: For alphanumeric only: `s = ''.join(c.lower() for c in s if c.isalnum())`

---

## 10. Variable Addition Output

### Problem Statement
Predict the output of basic variable addition in Python.

### Solution
```python
a = 5
b = 8

print(a + b)
```

### Expected Output
```
13
```

### Explanation
- Variables `a` and `b` are assigned integer values 5 and 8
- The `+` operator performs arithmetic addition on integers
- Result is 13, which is printed to console

### 💡 Pro Tips
- **Interview Tip**: This tests basic Python syntax and operator understanding
- **Type Considerations**: If these were strings, result would be concatenation: `'5' + '8' = '58'`
- **Dynamic Typing**: Python determines operation based on operand types at runtime

---

## 11. Extract Domain Name from Email

### Problem Statement
Given an email address, extract the domain name (the part between @ and the first dot).

### Solution

#### Method 1: Using Regular Expressions
```python
email = 'sundarpichai007@google.com'
import re

pattern = r'@(\w+)\.'
result = re.findall(pattern, email)
print("Domain using regex:", result)
```

#### Method 2: Using String Methods
```python
email = 'sundarpichai007@google.com'

# Split by @ and get the domain part
parts = email.split('@')
print("After splitting by @:", parts)

# Get domain name (before first dot)
domain = parts[-1].split('.')[0]
print("Domain name:", domain)
```

### Expected Output
```
Domain using regex: ['google']
After splitting by @: ['sundarpichai007', 'google.com']
Domain name: google
```

### Explanation

**Method 1 (Regex):**
- `@(\w+)\.` pattern matches @ followed by word characters, captured in group, followed by literal dot
- `\w+` matches one or more word characters (letters, digits, underscore)
- Parentheses create a capture group to extract just the domain name

**Method 2 (String Methods):**
- `email.split('@')` splits email into username and domain parts
- `parts[-1]` gets the last part (domain.extension)
- `.split('.')[0]` splits by dot and takes first part (domain name)

### 💡 Pro Tips
- **Regex Pros**: Powerful pattern matching, handles complex cases
- **String Methods Pros**: More readable, faster for simple cases, no imports needed
- **Interview Tip**: Discuss trade-offs between approaches
- **Edge Cases**: Handle subdomains like 'user@mail.google.com'
- **Production Code**: Consider using `email.split('@')[1].split('.')[0]` for simplicity

---

## 12. Extract Order ID Using Regular Expressions

### Problem Statement
Extract a specific order ID from a text string using regular expressions pattern matching.

### Solution
```python
str1 = 'The order id of a smartphone bought is ABC1234 and bcd5678'

import re

# Pattern to match: letter-letter-digit-digit-digit-digit
pattern = r'[a-zA-Z]{3}[0-9]{4}'

# Find all matches
res = re.findall(pattern, str1)
print("Order IDs found:", res)
```

### Expected Output
```
Order IDs found: ['ABC1234', 'bcd5678']
```

### Explanation
- `[a-zA-Z]{3}` matches exactly 3 letters (uppercase or lowercase)
- `[0-9]{4}` matches exactly 4 digits
- `re.findall()` returns all non-overlapping matches as a list
- Pattern captures order IDs in format: 3 letters + 4 digits

### 💡 Pro Tips
- **Alternative Pattern**: `r'[A-Za-z]{3}\d{4}'` (using `\d` for digits)
- **Case Sensitivity**: Use `re.IGNORECASE` flag for case-insensitive matching
- **Interview Tip**: Always ask about the exact format requirements
- **Validation**: Add word boundaries `\b` to avoid partial matches: `r'\b[a-zA-Z]{3}[0-9]{4}\b'`
- **Real-world**: Consider using `re.compile()` for repeated pattern matching

---

## 13. Swap Two Numbers

### Problem Statement
Swap the values of two variables without using a temporary variable.

### Solution

#### Method 1: Multiple Assignment (Pythonic)
```python
a = 10
b = 5

print(f"Before swap: a = {a}, b = {b}")

# Pythonic way - tuple unpacking
a, b = b, a

print(f"After swap: a = {a}, b = {b}")
```

#### Method 2: Traditional with Temporary Variable
```python
a = 10
b = 5

# Using temporary variable (commented approach from image)
# temp = a
# a = b
# b = temp
# print(a, b)
```

### Expected Output
```
Before swap: a = 10, b = 5
After swap: a = 5, b = 10
```

### Explanation
- **Tuple Unpacking**: `a, b = b, a` creates a tuple `(b, a)` and unpacks it to variables
- Python evaluates the right side first, then assigns to left side simultaneously
- No temporary variable needed - Python handles this internally
- Clean, readable, and efficient

### 💡 Pro Tips
- **Alternative Methods**:
  ```python
  # Arithmetic approach (works with numbers only)
  a = a + b
  b = a - b
  a = a - b
  
  # XOR approach (works with integers)
  a = a ^ b
  b = a ^ b
  a = a ^ b
  ```
- **Interview Tip**: Start with the Pythonic approach, then discuss alternatives
- **Edge Cases**: Arithmetic method can cause overflow with large numbers
- **Best Practice**: Always use tuple unpacking `a, b = b, a` in Python

---

## 16. Find the Length of the Longest Uninterrupted String

### Problem Statement
Given a string with multiple substrings separated by delimiters, find the length of the longest continuous substring.

### Solution
```python
a = 'xyz.xy.xyza'

# Split by delimiter and find max length
res = len(max(a.split('.'), key=len))
print("Length of longest substring:", res)

# Alternative: Show all substrings and their lengths
substrings = a.split('.')
print("All substrings:", substrings)
print("Lengths:", [len(s) for s in substrings])
print("Longest substring:", max(substrings, key=len))
```

### Expected Output
```
Length of longest substring: 4
All substrings: ['xyz', 'xy', 'xyza']
Lengths: [3, 2, 4]
Longest substring: xyza
```

### Explanation
- `a.split('.')` splits the string by delimiter '.' into a list of substrings
- `max(substrings, key=len)` finds the substring with maximum length using `len` as comparison key
- `len()` of the longest substring gives us the final answer
- The longest uninterrupted string is 'xyza' with length 4

### Alternative Solutions

#### Method 2: Manual iteration
```python
a = 'xyz.xy.xyza'

substrings = a.split('.')
max_length = 0

for substring in substrings:
    if len(substring) > max_length:
        max_length = len(substring)

print("Max length (manual):", max_length)
```

#### Method 3: Using map and max
```python
a = 'xyz.xy.xyza'

max_length = max(map(len, a.split('.')))
print("Max length (map):", max_length)
```

### 💡 Pro Tips
- **Different Delimiters**: Change `split('.')` to `split(delimiter)` for other separators
- **Multiple Delimiters**: Use regex for complex splitting: `re.split(r'[.,;]', string)`
- **Interview Tip**: Ask about edge cases - empty string, no delimiters, consecutive delimiters
- **Time Complexity**: O(n) where n is the length of the string
- **Space Complexity**: O(k) where k is the number of substrings created

---

## 17. Count the Number of Digits

### Problem Statement
Create a function to count the number of digits in any integer (positive or negative).

### Solution
```python
def count_digits(num):
    return len(str(abs(num)))

# Test the function
result = count_digits(453345)
print("Number of digits:", result)

# Test with negative number
result_negative = count_digits(-453345)
print("Number of digits (negative):", result_negative)
```

### Expected Output
```
Number of digits: 6
Number of digits (negative): 6
```

### Explanation
- `abs(num)` converts negative numbers to positive (removes minus sign)
- `str()` converts the number to string representation
- `len()` counts the characters in the string, which equals the number of digits
- Works for both positive and negative integers

### 💡 Pro Tips
- **Mathematical Approach**: 
  ```python
  import math
  def count_digits_math(num):
      if num == 0:
          return 1
      return math.floor(math.log10(abs(num))) + 1
  ```
- **Iterative Approach**:
  ```python
  def count_digits_loop(num):
      num = abs(num)
      count = 0
      while num > 0:
          count += 1
          num //= 10
      return count or 1  # Handle zero case
  ```
- **Interview Tip**: Discuss time complexity - String method: O(log n), Mathematical: O(1)
- **Edge Case**: Handle zero specially - it has 1 digit

---

## 18. Find the Second Largest Number

### Problem Statement
Given a list of numbers with possible duplicates, find the second largest unique number.

### Solution
```python
a = [12, 13, 45, 54, 34, 54, 12, 43]

# Remove duplicates using set, sort, and get second largest
res = sorted(set(a))[-2]
print("Second largest number:", res)

# Alternative approach with more steps for clarity
unique_numbers = list(set(a))
unique_numbers.sort()
second_largest = unique_numbers[-2]
print("Second largest (alternative):", second_largest)
```

### Expected Output
```
Second largest number: 45
Second largest (alternative): 45
```

### Explanation
- `set(a)` removes duplicate values: `{12, 13, 34, 43, 45, 54}`
- `sorted()` arranges unique numbers in ascending order: `[12, 13, 34, 43, 45, 54]`
- `[-2]` gets the second-to-last element (second largest)

### Alternative Solutions

#### Method 2: Using max() twice
```python
a = [12, 13, 45, 54, 34, 54, 12, 43]

largest = max(a)
# Remove all instances of largest number
filtered = [x for x in a if x != largest]
second_largest = max(filtered)
print("Second largest:", second_largest)
```

#### Method 3: Single pass algorithm
```python
def find_second_largest(arr):
    if len(arr) < 2:
        return None
    
    largest = second = float('-inf')
    
    for num in arr:
        if num > largest:
            second = largest
            largest = num
        elif num > second and num != largest:
            second = num
    
    return second if second != float('-inf') else None

a = [12, 13, 45, 54, 34, 54, 12, 43]
result = find_second_largest(a)
print("Second largest (single pass):", result)
```

### 💡 Pro Tips
- **Time Complexity**: 
  - Set + Sort method: O(n log n)
  - Single pass method: O(n) - more efficient for large datasets
- **Interview Tip**: Ask about edge cases - what if all numbers are the same?
- **Edge Cases**: 
  - Empty list or single element
  - All elements are identical
  - Only two unique elements
- **Space Optimization**: Single pass method uses O(1) extra space vs O(n) for set method

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
