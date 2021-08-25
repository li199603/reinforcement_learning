import multiprocessing as mp
import time 

def sender(pipe):
    for i in range(5):
        pipe.send(i)
        print("已发送%d" % (i))
    time.sleep(2)
    pipe.close()
        
def receiver(pipe):
    for i in range(10):
        if pipe.poll():
            print("can read")
            data = pipe.recv()
            print(data)
            time.sleep(1)
if __name__ == "__main__":
    pipe_ends = mp.Pipe()
    p1 = mp.Process(target=sender, args=(pipe_ends[0], ))
    p2 = mp.Process(target=receiver, args=(pipe_ends[1], ))
    for p in [p1, p2]:
        p.start()

    for p in [p1, p2]:
        p.join()