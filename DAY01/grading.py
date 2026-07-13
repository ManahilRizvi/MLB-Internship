def grading(average):
    if average>=86 and average<=100:
        grade="A"
        gp=4.0
        
    elif average>=82 and average<=85:
        grade="A-"
        gp=3.67
    
    elif average>=78 and average<=81:
        grade="B+"
        gp=3.33
        
    elif average>=74 and average<=77:
        grade="B"
        gp=3.00
        
    elif average>=70 and average<=73:
        grade="B-"
        gp=2.67
        
    elif average>=66 and average<=69:
        grade="C+"
        gp=2.33
    
    elif average>=62 and average<=65:
        grade="C"
        gp=2.00
    
    elif average>=58 and average<=61:
        grade="C-"
        gp=1.67
    
    elif average>=54 and average<=57:
        grade="D+"
        gp=1.33
    
    elif average>=50 and average<=53:
        grade="D"
        gp=1.00
        
    else:
        grade="F"
        gp=0.00
        
    return grade,gp
    
    
name=input("Enter Name: ")
semester=input("Enter Semester: ")
no_of_subjects=int(input("Enter No of Courses: " ))
courses=[]
scores=[]
print("...Please enter complete course name, Don't use abbrevitaions...")

for i in range(no_of_subjects):
    print("\nEnter Course", i+1)
    while True:
        course=input("Course Name: ")
        dup=False
        for c in courses:
            if course.lower()==c.lower():
                dup=True
                break
        if dup:
            print("ERROR! This course already exist.")
        else:
            break
    while True:
        marks=float(input("Marks: "))
        if marks<0:
            print("ERROR! Enter positive marks.")
        else:
            break
    courses.append(course)
    scores.append(marks)
    
total=sum(scores)
len_scores=len(scores)
avg=total/len_scores

print("Student Name: ", name)
print("Semester: ", semester)
print("\n---------Marks----------------")
for i in range(len(courses)):
    print(courses[i], ":", scores[i])

grades, cgpa=grading(avg)
print("\n---------Grades----------------")
print("Total Marks: ", total)
print("Average: ", avg)
print("Grade: ", grades)
print("CGPA: ", cgpa)
