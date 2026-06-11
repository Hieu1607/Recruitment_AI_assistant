import type { JobResponse } from "@/api";
import {
  Button,
  Modal,
  ModalContent,
  ModalDescription,
  ModalFooter,
  ModalHeader,
  ModalTitle,
} from "@/components/ui";
import { useEffect, useState } from "react";
import { fieldClasses } from "./job-utils";

interface DeleteJobDialogProps {
  job: JobResponse | null;
  open: boolean;
  loading?: boolean;
  onClose: () => void;
  onConfirm: () => void;
}

export function DeleteJobDialog({
  job,
  open,
  loading = false,
  onClose,
  onConfirm,
}: DeleteJobDialogProps) {
  const [confirmation, setConfirmation] = useState("");

  useEffect(() => {
    if (!open) setConfirmation("");
  }, [open, job?.id]);

  const canConfirm = job !== null && confirmation.trim() === job.title;

  return (
    <Modal open={open} onOpenChange={(nextOpen) => (!nextOpen ? onClose() : undefined)}>
      <ModalContent>
        <ModalHeader>
          <ModalTitle>Delete job</ModalTitle>
          <ModalDescription>
            This will remove the job workspace and its scoped data from normal access. This action
            cannot be undone.
          </ModalDescription>
        </ModalHeader>
        <div className="space-y-4">
          <p className="text-sm text-fg-muted">
            Type <span className="font-mono text-fg">{job?.title ?? ""}</span> to confirm.
          </p>
          <input
            value={confirmation}
            onChange={(event) => setConfirmation(event.target.value)}
            className={fieldClasses}
            placeholder={job?.title ?? ""}
            aria-label="Confirm job title"
          />
        </div>
        <ModalFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="danger" loading={loading} disabled={!canConfirm} onClick={onConfirm}>
            Delete job
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
