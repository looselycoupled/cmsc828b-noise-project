for f in mini.t2t.job.yaml baseline.100k.t2t.job.yaml baseline.200k.t2t.job.yaml baseline.300k.t2t.job.yaml
do
  yq w -i -d1 $f "spec.template.spec.containers[0].env.(name==AWS_ACCESS_KEY_ID).value" ""

  yq w -i -d1 $f "spec.template.spec.containers[0].env.(name==AWS_SECRET_ACCESS_KEY).value" ""
done

