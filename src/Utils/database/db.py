import sqlite3
import os
import  sys


# with open('horse.png','rb') as f:
#     m = f.read()
#     print(m) #testing to see if the binary format of the file is read.
    
#with open('prompt1.png','wb') as q:
#this saves horse.png as a new image as prompt1.png
    #q = f.write()
    
conn = sqlite3.connect('db')
cursor = conn.cursor()

#cursor.execute("""CREATE TABLE IF NOT EXISTS db_table (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, prompt TEXT, path BLOB)""")

with open('horse.png', 'rb') as f:
    data =f.read()
prompt = "man riding on a horse."

cursor.execute("""INSERT INTO db_table (prompt, path) VALUES (?,?)""",(prompt,data))
#put insert into as an if condition, select * if prompt exsists if doesnt then insert into and if exists then directly show with blob.

conn.commit()

m = cursor.execute("""SELECT * FROM db_table""")

for i in m:
    output = i[2]
    
with open('prompt.png' ,'wb') as f:
    f.write(output)

conn.commit()
cursor.close()
conn.close()

