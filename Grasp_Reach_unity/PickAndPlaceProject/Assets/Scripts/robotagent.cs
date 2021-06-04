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



public class robotagent : Agent
{
    [Header("Testing by DH")]
    [SerializeField]
    private GameObject[] armAxes;
    [SerializeField]
    private GameObject[] gripper;
    private Transform[] init_tran;
    private ArticulationBody[] init_arti;
    private bool check_robot = true ;
    public GameObject targetbox;

    public bool useVecObs = true ;
    public Planner_dh2 planner ;
    bool condition;
    
    bool condition2;
    

    
    
    //Rigidbody m_BallRb;
    //EnvironmentParameters m_ResetParams;
    // For Ros
    //public ROSConnection ros;
    //public string rosServiceName = "niryo_moveit";
    private float randX;
    private float randZ;

    public void Start()
    {
        /**
        for(int num = 0; num <armAxes.Length; num++)
        {
            init_tran[num] = armAxes[num].GetComponent<Transform>();

            init_arti[num] = armAxes[num].GetComponent<ArticulationBody>();
            
        }
        **/
    }
    
    
    
    public override void Initialize()
    {
        SetResetParameters();
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // if (useVecObs)
        // {
        //     sensor.AddObservation(gameObject.transform.rotation.z);
        //     sensor.AddObservation(gameObject.transform.rotation.x);
        //     sensor.AddObservation(ball.transform.position - gameObject.transform.position);
        //     sensor.AddObservation(m_BallRb.velocity);
        // }
        
        // Change Behavior param Also !
        if (useVecObs)
        {
            sensor.AddObservation(targetbox.transform.position.x);
            sensor.AddObservation(targetbox.transform.position.y);
            sensor.AddObservation(targetbox.transform.position.z);
        }


    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        
        /**
        var actionZ = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f);
        var actionX = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);

        if ((gameObject.transform.rotation.z < 0.25f && actionZ > 0f) ||
            (gameObject.transform.rotation.z > -0.25f && actionZ < 0f))
        {
            gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
        }

        if ((gameObject.transform.rotation.x < 0.25f && actionX > 0f) ||
            (gameObject.transform.rotation.x > -0.25f && actionX < 0f))
        {
            gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
        }
        if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
            Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
            Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
        {
            SetReward(-1f);
            EndEpisode();
        }
        else
        {
            SetReward(0.1f);
        }
        **/
        //check_robot = false;
        var action = actionBuffers.DiscreteActions[0];
        if (action == 1)
        {
            SetReward(0f);
            EndEpisode();
        }
        if (action == 2)
        {
            //planner.PublishJoints();
            for(int joint = 0;joint < armAxes.Length; joint++)
            {   
               
                var randt = (float)Random.Range(0f,2f);
                var jointx = armAxes[joint].GetComponent<ArticulationBody>().xDrive;
                jointx.target += randt;
                armAxes[joint].GetComponent<ArticulationBody>().xDrive = jointx;
                
                

            }
            SetReward(1f);
            check_robot = false;
            condition2 = true;
        }
        if (action == 3)
        {
            check_robot = false;
            planner.PublishJoints();

            
            
            SetReward(2f);
            
            condition = true;
            
        }
        if (action==4)
        {
            SetReward(-0.1f);
            check_robot = true;
        }
    }

    public override void OnEpisodeBegin()
    {

        SetResetParameters();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        //var continuousActionsOut = actionsOut.ContinuousActions;
        //continuousActionsOut[0] = -Input.GetAxis("Horizontal");
        //continuousActionsOut[1] = Input.GetAxis("Vertical");
    }

    public void Setrobot()
    {
        
        // Move to original pose
        /**
        for(int joint = 0;joint < armAxes.Length; joint++)
        {
            var init = init_arm[joint].GetComponent<ArticulationBody>().xDrive;

            //init.target = 0f;
            //init.targetvelocity = 0f;
            //init.angularVelocity = 0f;
            armAxes[joint].GetComponent<ArticulationBody>().xDrive = init;
        }
        
        for(int num = 0; num < armAxes.Length ; num++)
        {
            armAxes[num].GetComponent<Transform>().position = init_tran[num].position;
            armAxes[num].GetComponent<Transform>().rotation = init_tran[num].rotation;
            armAxes[num].GetComponent<ArticulationBody>().xDrive = init_arti[num].xDrive;



        }
        **/
        armAxes.All(c =>
        {
            
            c.transform.localRotation =  Quaternion.Euler(0f, 0f, 0f);
            c.transform.position =  new Vector3(0f, 0f, 0f);
            var t = c.GetComponent<ArticulationBody>().xDrive;
            t.target = 0f;
            c.GetComponent<ArticulationBody>().xDrive = t;
            return true;
        });
        /**
        gripper.All(c =>
        {
            var t = c.GetComponent<ArticulationBody>().xDrive;
            t.target = 0f;
            c.GetComponent<ArticulationBody>().xDrive = t;
            c.transform.localRotation =  Quaternion.Euler(0f, 0f, 0f);
            c.transform.localPosition =  new Vector3(0f, 0f, 0f);
            return true;
        });
        **/
        
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
        





        /**
        for(int joint = 0;joint < armAxes.Length; joint++)
        {
            var init = armAxes[joint].GetComponent<ArticulationBody>().xDrive;
            init.target = 0f;
            armAxes[joint].GetComponent<ArticulationBody>().xDrive = init;
        }
        **/
        


    }
    public void Settarget()
    {
        randZ = (float)Random.Range(0.18f,0.25f);
        randX = (float)Random.Range(0.2f,0.3f);
        //targetbox.transform.position = new UnityEngine.Vector3(0.2f,0.64f,0.2f);
        
        targetbox.transform.position = new UnityEngine.Vector3(randX,0.65f,randZ);
        //planner.target.transform.position = targetbox.transform.position;
        
    }
    public void SetResetParameters()
    {
        
        
        Setrobot();
        Settarget();
        check_robot = true;
        
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
        if(condition && !planner._waituntilTrue)
        {
            check_robot = true;
            condition = false;
        }
        if(condition2)
        {
            check_robot = true;
        }

    }


    
}
