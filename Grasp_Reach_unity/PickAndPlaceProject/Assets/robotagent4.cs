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


public class robotagent4 : Agent
{
    [Header("Testing by DH")]
    [SerializeField]
    private GameObject[] arms;
    [Header("Right,Left Order")]
    [SerializeField]
    private GameObject[] gripper;
    
    [Header("For the robot when get too far")]
    public GameObject servo_head;
    
    [HideInInspector]
    public Camera cam;

    [Header("For other settings")]
    public Planner_dh Planner_dh;


    [HideInInspector]
    public collision leftcollision;
    [HideInInspector]
    public collision rightcollision;
    [HideInInspector]
    public collision2 collision2;
    
    
    EnvironmentParameters m_ResetParams;
    public GameObject targetbox;
    private float[] action = new float[7];
    
    private bool check_robot = true ;
    public bool useVecObs = true ;  
    bool epi_start;    
    bool rightgripFlag ;
    bool leftgripFlag ;
    float closeThreshold = 0.046f;
    bool closeflag = false;
    bool gripperclosed;


    public void Start()
    {
        //SetResetParameters();
    }
    
    
    
    public override void Initialize()
    {
        SetResetParameters();
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        Planner_dh._waituntilTrue = false;
        
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
        // Num should be one Because arms[0] is shoulder link!
        if(gripperclosed || (!gripperclosed && !closeflag))
        {
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
            AddReward(0.02f);
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
        

        if(!Planner_dh._waituntilTrue)
        {
            Debug.Log("Publish");
            Debug.Log(targetbox.transform.position);
            Planner_dh.PublishJoints();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // TODO : 
       
    }

    public void Setrobot()
    {
 
        check_robot = false;
        arms.All(c =>
        {
            var t = c.GetComponent<ArticulationBody>().xDrive;
            t.target = 0f;
            c.GetComponent<ArticulationBody>().xDrive = t;
            c.transform.rotation =  Quaternion.Euler(0f, 0f, 0f);
            c.transform.position =  new Vector3(0f, 0f, 0f);
            
            return true;
        });
        

        gripper[0].transform.position = new Vector3(0f,0f,0f);
        gripper[0].transform.localRotation = Quaternion.Euler(0f,0f,0f);
        var rightx = gripper[0].GetComponent<ArticulationBody>().xDrive;
        rightx.target = 0f;
        gripper[0].GetComponent<ArticulationBody>().xDrive = rightx;

        
        gripper[1].transform.position = new Vector3(0f,0f,0f);
        gripper[1].transform.localRotation = Quaternion.Euler(0f,0f,0f);
        var leftx = gripper[1].GetComponent<ArticulationBody>().xDrive;
        leftx.target = 0f;
        gripper[1].GetComponent<ArticulationBody>().xDrive = leftx;

        
        


    }

    public void Settarget()
    {
        var randZ = Random.Range(0.2f,0.3f);
        var randX = Random.Range(-0.3f,0.3f);
        
        targetbox.transform.position = new UnityEngine.Vector3(randX,0.638f,randZ);
        targetbox.transform.rotation = Quaternion.Euler(0f,0f,0f);
    }

    public void SetParam()
    {
        check_robot = false;
        rightgripFlag = false;
        leftgripFlag = false;
        closeflag = false;
        gripperclosed = false;
    }

    public void SetResetParameters()
    {
        
        SetParam();
        
        Setrobot();
        Settarget();
        
        Debug.Log("Reset");
        Debug.Log(targetbox.transform.position);

    }

    public void FixedUpdate()
    {   

        var distance = servo_head.transform.TransformPoint(Vector3.zero).y - targetbox.transform.TransformPoint(Vector3.zero).y;
        //Debug.Log(distance);
       
        checkHeight();
        targetThreshold();
        Waitforunity();

        // TODO : Change as Agent 3
        rightgripFlag = false;
        leftgripFlag = false;      
         

    }

    void Waitforunity()
    {
        // TODO: Make condition
        if (check_robot)
        {
            RequestDecision();
        }
        
        if(!Planner_dh._waituntilTrue)
        {
            check_robot = true;
        }
        
        
    }
    

    private void OpenGripper()
    {
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
                SetReward(5f);
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

    public void TableHit()
    {
        
        AddReward(-1f);
        Debug.Log("Here");
 
        EndEpisode();
        
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


    private void WayTooFar()
    {
        
        var pos = servo_head.transform.TransformPoint(Vector3.zero).y;
        if (pos > 1.15f )
        {
            Debug.Log(pos);
            SetReward(-1f);
            EndEpisode();
        }
    }



}
