
start=1
#starting number
to=100
#range

for i in range(to) :
    fimename=""
    k=i+start
    if(k < 10) :
        filename="000"+str(k)+".txt"
    elif (k<100) :
        filename="00"+str(k)+".txt"
    elif (k<1000) :
        filename="0"+str(k)+".txt"
    f = open(filename, 'w')
    f.close()
