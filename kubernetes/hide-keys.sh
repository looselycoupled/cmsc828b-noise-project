for f in mini.t2t.job.yaml baseline.t2t.job.yaml
do
  yq w -i -d1 $f "spec.template.spec.containers[0].env.(name==AWS_ACCESS_KEY_ID).value" ""

  yq w -i -d1 $f "spec.template.spec.containers[0].env.(name==AWS_SECRET_ACCESS_KEY).value" ""
done

