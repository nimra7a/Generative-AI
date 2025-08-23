from typing import Optional
from pydantic import BaseModel, EmailStr, Field

#basic example
# class Student(BaseModel):
#   name : str

# new_student = {
#   'name' : 'Nimra Ansari'
# }

# student = Student(**new_student)
# print(student)
# print(type(student))

#default value
# class Student(BaseModel):
#   name : str = 'Nimra Ansari'

# new_student = {}

# student = Student(**new_student)
# print(student)
# print(type(student))

#optional field
class Student(BaseModel):
  name : str = 'Nimra Ansari'
  age : Optional[int] = None
  email : EmailStr
  cgpa : float = Field(gt=1.0, lt = 4.0, default= 3.84, description= "A decimal value representing the cgpa of student.")

new_student = {
  'age' : 23,
  'email' : 'abc@gmail.com' ,
  'cgpa' : 3.84
  }

student = Student(**new_student)
student_dict = dict(student)
print(student_dict)
print(type(student))

student_json = student.model_dump_json()
print(student_json)





