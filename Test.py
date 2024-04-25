from Exceptions import *
import re


class Animal:
    def __init__(self, name, age, species):
        self.name = name
        self.age = age
        self.species = species

    def eat(self, food):
        pass

    def make_sound(self):
        pass


class Lion(Animal):
    pregnant = False

    def __init__(self, name, age, species, mane_color, predatory, male):
        super().__init__(name, age, species)
        self.mane_color = mane_color
        self.predatory = predatory
        self.male = male

    def eat(self, food):
        print("The lion is eating " + food)

    def make_sound(self):
        print("Roar")

    def mate_with(self, lion):
        if self.male == lion.male:
            print("The same gender lions cannot be mated.")
        else:
            print("The lion " + self.name + "has mated with " + lion.name)
            if not self.male:
                self.pregnant = True
            else:
                lion.pregnant = True

    def give_birth(self):
        pass


class Parrot(Animal):
    vocabulary = []

    def __init__(self, name, age, species, feather_color):
        super().__init__(name, age, species)
        self.feather_color = feather_color

    def make_sound(self):
        print("Kya")

    def eat(self, food):
        print("The parrot eat " + food)

    def imitate_human_speech(self):
        pass

    def learn_word(self):
        pass


class Fish(Animal):

    def __init__(self, name, age, species, water_type, predatory, lay_eggs):
        super().__init__(name, age, species)
        self.water_type = water_type
        self.predatory = predatory
        self.lay_eggs = lay_eggs

    def make_sound(self):
        print("Buble-buble")

    def eat(self, food):
        print("The fish eat " + food)

    def swim(self):
        print("The fish is swimming")

    def lay_eggs(self, fish):
        pass


class Pet_shop:
    def __init__(self):

        self.animals = []

    def add_lion(self):
        while True:
            try:
                age = input("Input age: ")
                if age.isdigit():
                    age = int(age)
                    if age <= 0:
                        raise invalid_age
                    break
                else:
                    raise invalid_age
            except invalid_age:
                print("Enter a valid age.")
        name_pattern = re.compile("^[A-Z][-''a-z]{3,20}")
        while True:
            try:
                name = input("Input name: ")
                if name_pattern.match(name):
                    break
                else:
                    raise invalid_name
            except invalid_name:
                print("Enter a valid name.")
        colors = [
            "Red", "Green", "Blue", "Yellow", "Purple", "Orange", "Pink", "Brown",
            "Cyan", "Magenta", "Turquoise", "Lime", "Indigo", "Teal", "Maroon",
            "Olive", "Violet", "Gold", "Silver", "Gray"]
        while True:
            try:
                mane_color: str = input("Input mane color: ")
                mane = False
                for x in colors:
                    if mane_color == x:
                        mane = True
                if mane:
                    break
                else:
                    raise invalid_mane_color
            except invalid_mane_color:
                print("Input valid color.")
        while True:
            try:
                male = input("Input M for male and F for female: ")
                gender = None
                if str(male) == "M":
                    gender = True
                elif str(male) == "F":
                    gender = False
                else:
                    gender = None
                if gender == None:
                    raise invalid_gender
                else:
                    break
            except invalid_gender:
                print("Enter M or F.")

        lion = Lion(name, age, "Lion", mane_color, True, gender)
        self.animals.append(lion)

    def add_parrot(self):
        age = int(input("Input age: "))
        name = input("Input name: ")
        feather_color: str = input("Input feather color: ")
        parrot = Parrot(name, age, "Parrot", feather_color)
        self.animals.append(parrot)

    def add_fish(self):

        age = int(input("Input age: "))
        name = input("Input name: ")
        water_type: str = input("Input mane color: ")
        predatory = input("Input is it predatory (Y or N): ")
        ispredatory = None
        if str(predatory) == "Y":
            ispredatory = True
        else:
            ispredatory = False
        eggs = input("Input does it lay eggs (Y or N): ")
        lay_eggs = None
        if str(predatory) == "Y":
            lay_eggs = True
        else:
            lay_eggs = False
        fish = Fish(name, age, "Fish", water_type, ispredatory, lay_eggs)
        self.animals.append(fish)

    def lion_methods(self):
        lion_instance = None

        lions = []
        for x in self.animals:
            if x.species == "Lion":
                lions.append(x)
        if not lions:
            print("There are no lions so you cannot continue.")
            return
        else:
            lion_name = str(input("What is the name of the lion you are searching for: "))
            for x in lions:
                if x.name == lion_name:
                    lion_instance = x
            if lion_instance is None:
                print("There is no lion with such name.")
            while lion_instance is not None:
                print("1.Make sound")
                print("2.Eat")
                print("3.Mate with")
                print("4.Give birth")
                print("5.Exit")
                choice = int(input("Input your choice: "))
                match choice:
                    case 1:
                        lion_instance.make_sound()
                    case 2:
                        food = input("Input what food to eat: ")
                        lion_instance.eat(food)
                    case 3:
                        lions = []
                        for x in self.animals:
                            if x.species == "Lion":
                                lions.append(x)
                        for x in lions:
                            if len(lions) == 1:
                                print("There is only one or non lion`s so it cannot mate with anyone.")
                            else:
                                print("The available lions are: " + x.name)
                                name_of_lion = input("What is the name of the lion: ")
                                for x in lions:
                                    if x.name == name_of_lion:
                                        lion_to_mate = x

                                lion_instance.mate_with(lion_to_mate)
                    case 4:
                        lion_instance.give_birth()
                    case 5:
                        break


a = Pet_shop()
a.add_lion()
a.lion_methods()
