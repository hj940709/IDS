package com.hj.Hadoop;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.lib.MultipleInputs;
import org.apache.hadoop.mapreduce.Job;


public class Driver {

	@SuppressWarnings("rawtypes")
	public static void main(String[] args) throws IOException, 
	ClassNotFoundException, InterruptedException{
		// TODO Auto-generated method stub
		JobConf jobconf = new JobConf();
		Job job = Job.getInstance(jobconf, "Inner Join");
	    job.setJarByClass(Driver.class);
	    
	    MultipleInputs.addInputPath(jobconf, new Path(args[0]), TextInputFormat.class, (Class<? extends Mapper>) StudentMapper.class);
	    MultipleInputs.addInputPath(jobconf, new Path(args[1]), TextInputFormat.class, (Class<? extends Mapper>) ScoreMapper.class);
	    job.setReducerClass(JoinReducer.class);
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(Text.class);
	    
	    FileOutputFormat.setOutputPath(jobconf, new Path(args[2]));
	    System.exit(job.waitForCompletion(true)? 0 : -1);
	    //Fail to produce any result
	    //Keep reporting output directory is not set, which is wired
	    //No plan to debug unless solid answer has been released
	}

}
