# 📘 Python Operators Reference Guide

This document contains all important Python operators, their names, meanings, and usage examples.

---

# 🔹 1️⃣ Arithmetic Operators

Used for mathematical calculations.

| Symbol | Name | Example | Output | Why We Use |
|--------|------|----------|--------|------------|
| `+` | Addition | `5 + 2` | 7 | To add values |
| `-` | Subtraction | `5 - 2` | 3 | To subtract values |
| `*` | Multiplication | `5 * 2` | 10 | Scaling numbers |
| `/` | Division | `5 / 2` | 2.5 | Exact division (returns float) |
| `//` | Floor Division | `5 // 2` | 2 | Integer division (removes decimal) |
| `%` | Modulus | `5 % 2` | 1 | Get remainder (even/odd check) |
| `**` | Exponent | `2 ** 3` | 8 | Power calculation |

---

# 🔹 2️⃣ Comparison Operators

Used to compare values. Returns `True` or `False`.

| Symbol | Name | Example | Why We Use |
|--------|------|----------|------------|
| `==` | Equal to | `5 == 5` | Check equality |
| `!=` | Not equal | `5 != 3` | Check inequality |
| `>` | Greater than | `5 > 3` | Filtering / conditions |
| `<` | Less than | `3 < 5` | Filtering / conditions |
| `>=` | Greater than or equal | `5 >= 5` | Range checking |
| `<=` | Less than or equal | `3 <= 5` | Range checking |

---

# 🔹 3️⃣ Logical Operators

Used to combine multiple conditions.

| Operator | Example | Meaning |
|----------|----------|----------|
| `and` | `x > 5 and x < 10` | Both conditions must be true |
| `or` | `x > 5 or x < 2` | At least one condition true |
| `not` | `not True` | Reverses the condition |

---

# 🔹 4️⃣ Assignment Operators

Used to assign values to variables.

| Operator | Example | Meaning |
|----------|----------|----------|
| `=` | `x = 5` | Assign value |
| `+=` | `x += 2` | Add and assign |
| `-=` | `x -= 2` | Subtract and assign |
| `*=` | `x *= 2` | Multiply and assign |
| `/=` | `x /= 2` | Divide and assign |
| `//=` | `x //= 2` | Floor divide and assign |
| `%=` | `x %= 2` | Modulus and assign |
| `**=` | `x **= 2` | Power and assign |

---

# 🔹 5️⃣ Membership Operators

Used to check whether a value exists in a sequence.

| Operator | Example | Meaning |
|----------|----------|----------|
| `in` | `"a" in "apple"` | Returns True if present |
| `not in` | `"x" not in "apple"` | Returns True if not present |

Commonly used with:
- Lists
- Strings
- Tuples
- Sets
- Dictionaries

---

# 🔹 6️⃣ Identity Operators

Used to compare memory locations.

| Operator | Example | Meaning |
|----------|----------|----------|
| `is` | `x is y` | True if same object |
| `is not` | `x is not y` | True if different objects |

⚠ Note: `is` is different from `==`

---

# 🔹 7️⃣ Bitwise Operators (Advanced)

Used in low-level programming.

| Operator | Name |
|----------|------|
| `&` | AND |
| `|` | OR |
| `^` | XOR |
| `~` | NOT |
| `<<` | Left Shift |
| `>>` | Right Shift |

---

# 🔹 8️⃣ Special Symbols in Python

| Symbol | Usage |
|--------|--------|
| `:` | Used in functions, loops, if statements |
| `()` | Function calls / grouping |
| `[]` | Lists / indexing |
| `{}` | Dictionaries / sets |
| `#` | Comment |
| `@` | Decorator |
| `_` | Temporary variable |
| `\` | Escape character |

---

# 🔥 Important Formula

Python follows this identity:

