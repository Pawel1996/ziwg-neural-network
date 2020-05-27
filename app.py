import ann.ann_runnable
import sys
if __name__ == "__main__":
        if len(sys.argv) != 3:
            print("usage python app.py <src> <dest>") 
        else:
            ann.ann_runnable.start(sys.argv[1], sys.argv[2], [16], 0.4, [3,5], [10])

