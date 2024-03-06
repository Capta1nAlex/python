board = []
rows=""
columns=""
fields=""

print("Enter 9 lines:")
for i in range(9):
    user_input = input(f"{i+1}: ")
    board.append(user_input)

for i in range(9):
    for j in board:
        if (rows.find(j[i])!=-1):
            print("Sudoku is wrong!")
            exit()
        rows+=j[i]
    print(rows)
    rows=""
    
for i in range(9):
    for j in board[i]:
        print(j)
        if (columns.find(j)!=-1):
           print("Sudoku is wrong!")
           exit()
        columns+=j
    print(columns)
    columns=""

for i in range(0, 9, 3):
    for j in range(0, 9, 3):
        fields = [board[row][col] for row in range(i, i+3) for col in range(j, j+3)]
        fields=''.join(fields)
        print(fields)
        for k in fields:        
            if (fields.count(k)>1):
               print("Sudoku is wrong!")
               exit()
    fields=""
print("Sudoku is solved!")