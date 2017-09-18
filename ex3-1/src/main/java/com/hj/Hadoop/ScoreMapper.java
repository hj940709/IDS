package com.hj.Hadoop;

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class ScoreMapper extends Mapper<Text, Text, Text, Text> {

	@Override
	protected void map(Text key, Text value, Mapper<Text, Text, Text, Text>.Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		String[] line = value.toString().split(",");
		if(Integer.valueOf(line[1])>80 && Integer.valueOf(line[2])<=95) {
			String newkey = line[0];
			String values = "score,"+line[1]+","+line[2]+","+line[3];
			context.write(new Text(newkey), new Text(values));
		}
	}

}
