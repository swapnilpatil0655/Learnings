def welcome():
    print("Welcome to Taj\n")
    menu()

def menu():
    print("Press 1 for veg")
    print("Press 2 for non-veg")
    print("Press 3 for bill")
    print("Press 4 to exit\n")
    opt = int(input())
    if opt == 1:
        veg_menu()
    elif opt == 2:
        non_veg_menu()
    elif opt == 3:
        calculate_bill()
    elif opt == 4:
        print("Thank you for visiting Taj")
    else:
        print("Invalid option")
        menu()

def veg_menu():
    print("Veg dishes")
    print("Press 1 for mix-veg")
    print("Press 2 for dal")
    print("Press 3 for rice\n")
    veg_opt = int(input())
    if veg_opt == 1:
        mix_veg()
    elif veg_opt == 2:
        dal()
    elif veg_opt == 3:
        rice()
    else:
        print("Invalid option")
        veg_menu()

def non_veg_menu():
    print("Non-veg dishes")
    print("Press 1 for chicken")
    print("Press 2 for biryani")
    print("Press 3 for shawarma\n")
    non_veg_opt = int(input())
    if non_veg_opt == 1:
        chicken()
    elif non_veg_opt == 2:
        biryani()
    elif non_veg_opt == 3:
        shawarma()
    else:
        print("Invalid option")
        non_veg_menu()

def mix_veg():
    print("Mix-veg selected")
    qty = int(input("Enter quantity: "))
    price = 50
    total = qty * price
    print("Mix-veg qty =", qty, "price =", price, "Total =", total)
    menu()

def dal():
    print("Dal selected")
    qty = int(input("Enter quantity: "))
    price = 60
    total = qty * price
    print("Dal qty =", qty, "price =", price, "Total =", total)
    menu()

def rice():
    print("Rice selected")
    qty = int(input("Enter quantity: "))
    price = 30
    total = qty * price
    print("Rice qty =", qty, "price =", price, "Total =", total)
    menu()

def chicken():
    print("Chicken selected")
    qty = int(input("Enter quantity: "))
    price = 120
    total = qty * price
    print("Chicken qty =", qty, "price =", price, "Total =", total)
    menu()

def biryani():
    print("Biryani selected")
    qty = int(input("Enter quantity: "))
    price = 150
    total = qty * price
    print("Biryani qty =", qty, "price =", price, "Total =", total)
    menu()

def shawarma():
    print("Shawarma selected")
    qty = int(input("Enter quantity: "))
    price = 80
    total = qty * price
    print("Shawarma qty =", qty, "price =", price, "Total =", total)
    menu()

def calculate_bill():
    total = 0
    print("Enter the number of each dish you ordered")
    qty = int(input("Mix-veg: "))
    total += qty * 50
    qty = int(input("Dal: "))
    total += qty * 60
    qty = int(input("Rice: "))
    total += qty * 30
    print("Your total is: ",total)
  
welcome()
   
