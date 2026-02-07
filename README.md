# world2data

Generate ground truth data from videos. 

Creates a 3D map of containing 

- coordinates with item labels
- action hints like "graspable" for "doorhandle"
- state changes like "door opened" "cup vanished" 

Reflects on improbable state changes that cannot be explained by an items motion model:

"Door opened" -> OK
"Cup vanished" -> a reasoning model is asked to understand why -> "the person who just drank from the cup was seen leaving the room" 
"

## Interfaces 

- Camera motion from video
- Object detection (bounding boxes or even segmention) 
- Depth estimation from video 
- Cost function for particles 
    - This can include the output from the depth estimation, camera from motion, object detection 
    - Should be agnostic to what exactly the interfaces give. e.g. object detection gives bounding boxes or segmentation - works with both 
- Motion model for each object: Maps current particle position in state space to new position + spread. The motion model may change based on the context: 
    - An object in someones hand is expected to move 
    - Objects on the table are not expected to move (they move with the table top)

- State change feedback
    - If a motion happens that is not explained adequatly by the model (things vanish, things that are not supposed to move, move), this information is fed into a reasoning model which can explain why 
    - The output from the reasoning model is used to adjust the motion model

All tied together with a particle filter:

Each *new* object detection spawns a new particle. Motion model and particle evolution is applied, then particles are sampled using cost function. 

## Idea


LFM2.5‑VL generates scene description (structured)
    - what objects this scene contains 
    - relationships "person drinks from mug" 

SLAM first to get camera motion from images and set a coordinate frame
Use Yolo + 3D estimation to find objects 
map them to 3D world (SLAM) coordinates 
find the objects mentioned by LFM before 

track each item using a particle filter 
LFM suggests a motion model for each particle 

Correlate the things found by Yolo with the object descriptions. 

The output should be
    - list of items that exist in the entire video and where in 3D space (initial condition)
    - list of actions that happen e.g.
        - "door opens at time t" (e.g. from LFM)
        - "cup vanishes at time t" (should be there according to PF but isn't found)
        - "person leaves room at time t" (e.g. from LFM)

We now have a graph and can look up items which are associated with unexplainable things, like the cup vanishing. The cup node is linked to the person node, we can collect all the cups actions and persons actions from the graph and get 


"cup seen first at x,y,z" 
"person drinks from cup at time t" 
"person leaves the room"
"cup expected at x,y,z but not found" 


## Resources 

https://arxiv.org/html/2602.04517v1
https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B?utm_source=chatgpt.com
https://rerun.io/