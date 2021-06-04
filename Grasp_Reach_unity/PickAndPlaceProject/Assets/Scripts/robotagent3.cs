using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections;
using System.Linq;

////Define Grasp Task


public class robotagent3 : Agent
{
    [Header("Testing by DH")]
    [SerializeField]
    private GameObject[] arms;
    [Header("Right,Left Order")]
    [SerializeField]
    private GameObject[] gripper;
    
    [Header("For the robot when get too far")]
    public GameObject servo_head;

    public Camera cam;


    private ArticulationBody articulationbody;


    [HideInInspector]
    public collision leftcollision;
    [HideInInspector]
    public collision rightcollision;
    [HideInInspector]
    public collision2 collision2;
    
    private bool check_robot = true ;
    public GameObject targetbox;
    
    // public Camera camera;
    
    private float[] action = new float[7];
    
    
    public bool useVecObs = true ;
    
    
    //public Planner_dh2 planner ;
    
    EnvironmentParameters m_ResetParams;
    
    bool rightgripFlag ;
    bool leftgripFlag ;
    float closeThreshold = 0.05f;
    bool closeflag = false;
    bool gripperclosed;


    public void Start()
    {
        
    }
    
    
    
    public override void Initialize()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        SetResetParameters();
        //Debug.Log(m_ResetParams.GetWithDefault());
        
    }
   
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        
        action[0] = Mathf.Clamp(actionBuffers.ContinuousActions[0],-3f,3f);
        action[1] = Mathf.Clamp(actionBuffers.ContinuousActions[1],-3f,3f);
        action[2] = Mathf.Clamp(actionBuffers.ContinuousActions[2],-3f,3f);
        action[3] = Mathf.Clamp(actionBuffers.ContinuousActions[3],-3f,3f);
        action[4] = Mathf.Clamp(actionBuffers.ContinuousActions[4],-3f,3f);
        // For Gripper
        action[5] = Mathf.Clamp(actionBuffers.ContinuousActions[5],-0.01f,0.01f);
        action[6] = Mathf.Clamp(actionBuffers.ContinuousActions[6],-0.01f,0.01f);
        
        
        
        // Gripper not closed & CloseFlag Ok
        if(!gripperclosed && closeflag )
        {
            var rightx = gripper[0].GetComponent<ArticulationBody>().xDrive;
            rightx.target = action[5];
            gripper[0].GetComponent<ArticulationBody>().xDrive = rightx;
            
            var leftx = gripper[1].GetComponent<ArticulationBody>().xDrive;
            leftx.target = action[6];
            gripper[1].GetComponent<ArticulationBody>().xDrive = leftx;
            Debug.Log("Gripper Closed");
            gripperclosed = true;
        }
        
        // Gripper closed or Close flag False & Gripper open
        if(gripperclosed || (!gripperclosed && !closeflag))
        {
            // Num should be one Because arms[0] is shoulder link!
            // Baselink move,..?
            for(int num = 1; num <arms.Length ; num++)
            {
                var armX = arms[num].GetComponent<ArticulationBody>().xDrive;
                armX.target += action[num-1];
                arms[num].GetComponent<ArticulationBody>().xDrive = armX;
                
            }
        }
        WayTooFar();
        AddReward(-0.01f);
        
        if (rightgripFlag && leftgripFlag)
        {
            AddReward(0.2f);
        }



        // If we give distance as reward robot might just stand near and do nothing
        var distance0 = Vector3.Distance(targetbox.transform.TransformPoint(Vector3.zero).normalized, gripper[0].transform.TransformPoint(Vector3.zero).normalized);
        var distance1 = Vector3.Distance(targetbox.transform.TransformPoint(Vector3.zero).normalized, gripper[1].transform.TransformPoint(Vector3.zero).normalized);
        var distance = (distance0 + distance1)/2;
        AddReward(-distance);
    
            
    }

    public override void OnEpisodeBegin()
    {

        SetResetParameters();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // TODO : 
       
    }

    public void Setrobot()
    {
        
        check_robot = false;

        // Shoulder
        var articulxDrive = arms[0].GetComponent<ArticulationBody>().xDrive;
        //articulxDrive.target = 3.218f;
        articulxDrive.target = m_ResetParams.GetWithDefault("shoulder",3.218f);
        arms[0].GetComponent<ArticulationBody>().xDrive = articulxDrive;
        arms[0].transform.position = new Vector3(0f,0f,0f);
        arms[0].transform.localRotation =  Quaternion.Euler(0f, 0f, 0f);
        


        // Arm
        articulxDrive = arms[1].GetComponent<ArticulationBody>().xDrive;
        articulxDrive.target = m_ResetParams.GetWithDefault("arm",-61.4f);
        arms[1].GetComponent<ArticulationBody>().xDrive = articulxDrive;
        arms[1].transform.position = new Vector3(0f,0f,0f);
        arms[1].transform.localRotation =  Quaternion.Euler(0f, 0f, 0f);
        
        // Elbow
        articulxDrive = arms[2].GetComponent<ArticulationBody>().xDrive;
        articulxDrive.target = m_ResetParams.GetWithDefault("elbow",10.66f);
        arms[2].GetComponent<ArticulationBody>().xDrive = articulxDrive;
        arms[2].transform.position = new Vector3(0f,0f,0f);
        arms[2].transform.localRotation =  Quaternion.Euler(0f, 0f, 0f);
        
        // Forearm
        articulxDrive = arms[3].GetComponent<ArticulationBody>().xDrive;
        articulxDrive.target = m_ResetParams.GetWithDefault("forearm",-0.059f);
        arms[3].GetComponent<ArticulationBody>().xDrive = articulxDrive;
        arms[3].transform.position = new Vector3(0f,0f,0f);
        arms[3].transform.localRotation =  Quaternion.Euler(0f, 0f, 0f);
        
        // Wrist
        articulxDrive = arms[4].GetComponent<ArticulationBody>().xDrive;
        articulxDrive.target = m_ResetParams.GetWithDefault("wrist",-39.21f);
        arms[4].GetComponent<ArticulationBody>().xDrive = articulxDrive;
        arms[4].transform.position = new Vector3(0f,0f,0f);
        arms[4].transform.localRotation =  Quaternion.Euler(0f, 0f, 0f);
        
        // Hand
        articulxDrive = arms[5].GetComponent<ArticulationBody>().xDrive;
        articulxDrive.target = m_ResetParams.GetWithDefault("hand",95.01f);
        arms[5].GetComponent<ArticulationBody>().xDrive = articulxDrive;
        arms[5].transform.position = new Vector3(0f,0f,0f);
        arms[5].transform.localRotation =  Quaternion.Euler(0f, 0f, 0f);
        
        
        // Gripper Right Left
        gripper[0].transform.position = new Vector3(0f,0f,0f);
        gripper[0].transform.localRotation = Quaternion.Euler(0f,0f,0f);
        var rightx = gripper[0].GetComponent<ArticulationBody>().xDrive;
        rightx.target = m_ResetParams.GetWithDefault("right_grip",0f);
        gripper[0].GetComponent<ArticulationBody>().xDrive = rightx;

        
        gripper[1].transform.position = new Vector3(0f,0f,0f);
        gripper[1].transform.localRotation = Quaternion.Euler(0f,0f,0f);
        var leftx = gripper[1].GetComponent<ArticulationBody>().xDrive;
        leftx.target = m_ResetParams.GetWithDefault("left_grip",0f);
        gripper[1].GetComponent<ArticulationBody>().xDrive = leftx;
        
        OpenGripper();

        

        


    }
    public void Settarget()
    {
        
        targetbox.transform.position = new UnityEngine.Vector3(-0.03f,0.638f,0.341f);
        targetbox.transform.rotation = Quaternion.Euler(0f,0f,0f);
    }
    public void SetResetParameters()
    {
        
        
        Setrobot();
        Settarget();
        
        
        rightgripFlag = false;
        leftgripFlag = false;
        closeflag = false;
        gripperclosed = false;
        check_robot = true;
        
    }

    public void FixedUpdate()
    {   

        var distance = servo_head.transform.TransformPoint(Vector3.zero).y - targetbox.transform.TransformPoint(Vector3.zero).y;
        //Debug.Log(distance);
       
        checkHeight();
        targetThreshold();
        Waitforunity();

        if (cam != null)
        {
            cam.Render();
        }
       
         

    }

    void Waitforunity()
    {
        // TODO: Make condition
        if (check_robot)
        {
            RequestDecision();
        }
        /**
        if(condition && !planner._waituntilTrue)
        {
            check_robot = true;
            condition = false;
        }
        
        if(condition2)
        {
            check_robot = true;
        }
        **/

    }

    public void TableHit()
    {
        AddReward(-1f);
        EndEpisode();
    }
    

    private void OpenGripper()
    {
        gripper[0].transform.position = new Vector3(0f,0f,0f);
        gripper[1].transform.position = new Vector3(0f,0f,0f);
        var right = gripper[0].GetComponent<ArticulationBody>().xDrive;
        var left = gripper[1].GetComponent<ArticulationBody>().xDrive;
        
        right.target = 0.01f;
        left.target = -0.01f;

        gripper[0].GetComponent<ArticulationBody>().xDrive = right;
        gripper[1].GetComponent<ArticulationBody>().xDrive = left;

    }

    private void checkHeight()
    {
        if(gripperclosed && rightgripFlag && leftgripFlag)
        {
            // TODO : Check the target box grasp success
            var targetHeight = targetbox.GetComponent<Transform>().position.y;
            if(targetHeight > 0.7f)
            {
                SetReward(10f);
                EndEpisode();
            }
            
            
        }

    }

    private void targetThreshold()
    {
        var distance = servo_head.transform.TransformPoint(Vector3.zero).y - targetbox.transform.TransformPoint(Vector3.zero).y;
        if(distance < closeThreshold)
        {
            closeflag = true;
        }
    }

    public void RobotHit()
    {
        SetReward(-1f);
        EndEpisode();
    }

    public void LeftHit()
    {
        leftgripFlag = true;
    }
    
    public void RightHit()
    {
        rightgripFlag = true;
    }

    public void LeftHitCancel()
    {
        leftgripFlag = false;
    }


    public void RightHitCancel()
    {
        rightgripFlag = false;
    }

    private void WayTooFar()
    {
        
        var pos = servo_head.transform.TransformPoint(Vector3.zero).y;
        if (pos >1f )
        {
            Debug.Log(pos);
            SetReward(-1f);
            EndEpisode();
        }
    }



}
