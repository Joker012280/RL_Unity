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

//// Define Reach Target Task

public class robotagent2 : Agent
{
    [Header("Testing by DH")]
    [SerializeField]
    private GameObject[] arms;
    [Header("Right,Left Order")]
    [SerializeField]
    private GameObject[] gripper;
    

    [HideInInspector]
    public collision collision;
    [HideInInspector]
    public collision2 collision2;
    private float[] rotations = new float[6]; // 6 means num of arms
    private bool check_robot = true ;
    public GameObject targetbox;
    private float[] action = new float[6];
    public bool useVecObs = true ;
    public Planner_dh2 planner ;
    bool condition;
    
    bool condition2;
  
    private float randX;
    private float randZ;

    public void Start()
    {
  
    }
    
    
    
    public override void Initialize()
    {
        SetResetParameters();
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {
  
        var distance0 = Vector3.Distance(targetbox.transform.TransformPoint(Vector3.zero).normalized, gripper[0].transform.TransformPoint(Vector3.zero).normalized);
        var distance1 = Vector3.Distance(targetbox.transform.TransformPoint(Vector3.zero).normalized, gripper[1].transform.TransformPoint(Vector3.zero).normalized);
        var distance = (distance0 + distance1)/2;
        // Change Behavior param Also !
        if (useVecObs)
        {
            sensor.AddObservation(targetbox.transform.TransformPoint(Vector3.zero).normalized);
            sensor.AddObservation((arms[0].transform.rotation.y)/180f);
            sensor.AddObservation((arms[1].transform.rotation.x)/180f);
            sensor.AddObservation((arms[2].transform.rotation.y)/180f);
            sensor.AddObservation((arms[3].transform.rotation.x)/180f);
            sensor.AddObservation((arms[4].transform.rotation.x)/180f);
            sensor.AddObservation((arms[5].transform.rotation.x)/180f);
            sensor.AddObservation((gripper[0].transform.TransformPoint(Vector3.zero).normalized));
            sensor.AddObservation((gripper[1].transform.TransformPoint(Vector3.zero).normalized));
            sensor.AddObservation(distance);
        
        }


    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        
        action[0] = Mathf.Clamp(actionBuffers.ContinuousActions[0],-3f,3f);
        action[1] = Mathf.Clamp(actionBuffers.ContinuousActions[1],-3f,3f);
        action[2] = Mathf.Clamp(actionBuffers.ContinuousActions[2],-3f,3f);
        action[3] = Mathf.Clamp(actionBuffers.ContinuousActions[3],-3f,3f);
        action[4] = Mathf.Clamp(actionBuffers.ContinuousActions[4],-3f,3f);
        action[5] = Mathf.Clamp(actionBuffers.ContinuousActions[5],-3f,3f);
        for(int num = 0; num <arms.Length ; num++)
        {
            var armX = arms[num].GetComponent<ArticulationBody>().xDrive;
            armX.target += action[num];
            arms[num].GetComponent<ArticulationBody>().xDrive = armX;
        }
        AddReward(-0.01f);
        var distance0 = Vector3.Distance(targetbox.transform.TransformPoint(Vector3.zero).normalized, gripper[0].transform.TransformPoint(Vector3.zero).normalized);
        var distance1 = Vector3.Distance(targetbox.transform.TransformPoint(Vector3.zero).normalized, gripper[1].transform.TransformPoint(Vector3.zero).normalized);
        var distance = (distance0 + distance1)/2;
        AddReward(-distance);
    
    }

    public override void OnEpisodeBegin()
    {

        SetResetParameters();
        check_robot =true;
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
            
            c.transform.localRotation =  Quaternion.Euler(0f, 0f, 0f);
            c.transform.position =  new Vector3(0f, 0f, 0f);
            var t = c.GetComponent<ArticulationBody>().xDrive;
            t.target = 0f;
            c.GetComponent<ArticulationBody>().xDrive = t;
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
    public void Settarget() // Fixed location and random rotation
    {
        randZ = (float)Random.Range(0.15f,0.25f);
        randX = (float)Random.Range(-0.05f,0.4f);
        
        targetbox.transform.position = new UnityEngine.Vector3(randX,0.65f,randZ);
        
    }
    public void SetResetParameters()
    {
        
        
        Setrobot();
        Settarget();
        
        
    }

    public void FixedUpdate()
    {   


        Waitforunity();

    }

    void Waitforunity()
    {
        // TODO: Make condition
        if (check_robot)
        {
            RequestDecision();
        }
        
        

    }

    public void TableHit()
    {
        if(check_robot)
        {
            AddReward(-1f);
            EndEpisode();
        }
    }
    
    public void TargetHit()
    {
        AddReward(10f);
        Settarget();
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.transform.CompareTag("table"))
        {
            TableHit();
            Debug.Log("Table");
        }
        else if (other.transform.CompareTag("target"))
        {
            TargetHit();
            Debug.Log("Target");
        }
        else
        {
            Debug.LogError("I Don't know");
        }
    }


    
}
