import cv2
import numpy as np
from pathlib import Path
from collections import deque

gridsize=40
cellsize=20
mod=10**9+7
maze=Path("mazes")
answer=Path("answers")
answer.mkdir(exist_ok=True)
moves=[(-1,0),(1,0),(0,-1),(0,1)]

def iswall(img,row,col):
    y=row*cellsize
    x=col*cellsize
    margin=cellsize//6
    interior=img[
        y+margin:y+cellsize-margin,
        x+margin:x+cellsize-margin
    ]
    b=interior[:,:,0]
    r=interior[:,:,2]
    red_mask=(r>150)&(b<120)
    wall_pixels=np.count_nonzero(red_mask)
    return wall_pixels>(red_mask.size//2)

def extract(img):
    grid=[[False]*gridsize for _ in range(gridsize)]

    for r in range(gridsize):
        for c in range(gridsize):
            grid[r][c]=iswall(img,r,c)
    
    return grid

def bfs(grid):
    start=(0,0)
    goal=(gridsize-1,gridsize-1)

    if grid[0][0] or grid[gridsize-1][gridsize-1]:
        print(f"Skipping.")
        return -1,[]
    
    q=deque([start])
    parent={start:None}
    dist={start:1}
    while q:
        r,c=q.popleft()

        if (r,c)==goal:
            break

        for dr,dc in moves:
            nr,nc=r+dr,c+dc

            if not (0<=nr<gridsize and 0<=nc<gridsize):
                continue
            
            if grid[nr][nc]:
                continue
            
            if (nr,nc) in parent:
                continue
            
            parent[(nr,nc)]=(r,c)
            dist[(nr,nc)]=dist[(r,c)]+1
            q.append((nr,nc))
    
    if goal not in parent:
        return -1,[]
    
    path=[]
    cur=goal
    while cur is not None:
        path.append(cur)
        cur=parent[cur]
    path.reverse()

    return dist[goal],path

def draw(img,path):
    out=img.copy()
    points=[]

    for r,c in path:
        cx=c*cellsize+cellsize//2
        cy=r*cellsize+cellsize//2
        points.append((cx,cy))
    
    for i in range(len(points)-1):
        cv2.line(out,points[i],points[i+1],(0,255,0),4)
    
    for p in points:
        cv2.circle(out,p,3,(0,255,0),-1)
    
    return out

def impossible():
    img=np.zeros((800,800,3),dtype=np.uint8)
    img[:]=(255,0,0)
    cv2.putText(
        img,
        "IMPOSSIBLE",
        (70, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3
    )
    return img

def main():
    maze_files=sorted(
        maze.glob("*.png"),
        key=lambda p:int(p.stem.split("_")[1])
    )
    print(f"Found {len(maze_files)} mazes to solve.")

    password=1
    solved=0
    impossible_count=0
    print("Solving mazes...")
    for maze_file in maze_files:
        
        img=cv2.imread(str(maze_file))
        if img is None:
            print(f"Failed to read file {maze_file.name}")
            continue

        kernel=np.ones((5,5),dtype=np.uint8)
        img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
        grid=extract(img)
        length,path=bfs(grid)

        if length==-1:
            impossible_count+=1
            out=impossible()
            print(f"{maze_file.name}: maze impossible to solve.")
        else:
            solved+=1
            password=(password*length)%mod
            out=draw(img,path)
            print(f"{maze_file.name}: Steps={length}")

        out_path=answer/maze_file.name
        cv2.imwrite(str(out_path),out)

    print(
        f"Solved: {solved}\n"
        f"Impossible to solve: {impossible_count}\n"
        f"Total: {solved+impossible_count}\n"
        f"Final Password: {password}"
    )

    if solved+impossible_count!=100:
        print(
            f"Warning: processed only "
            f"{solved+impossible_count}/100 files."
        )

    with open("password.txt","w") as f:
        f.write(str(password))
    print("password.txt generated.")


if __name__ == "__main__":
    main()