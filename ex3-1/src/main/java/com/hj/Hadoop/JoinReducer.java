package com.hj.Hadoop;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class JoinReducer extends Reducer<Text, Text, Text, Text> {

	@Override
	protected void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		Iterator<Text> iterator =  values.iterator();
		String student = "";
		String score = "";
		while(iterator.hasNext()) {
			String current_value = iterator.next().toString();
			String identifier = current_value.split(",")[0];
			if("student".equals(identifier))
				student = current_value;
			else if("score".equals(identifier))
				score += current_value+",";
		}
		score = score.substring(0, score.length()-1);
		String reduced_value = student+","+score;
		context.write(key, new Text(reduced_value));
	}
	
}
